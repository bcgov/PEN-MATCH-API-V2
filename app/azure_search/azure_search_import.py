import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.core.exceptions import HttpResponseError
from openai import AzureOpenAI

from database.postgresql import PostgreSQLManager
from config.settings import settings


@dataclass
class AzureSearchProcessingStats:
    total_processed: int = 0
    total_failed: int = 0
    start_time: float = 0
    batches_completed: int = 0


class AzureSearchImportService:
    def __init__(self, max_db_connections: int = 20):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"

        # Azure Search client
        self.credential = DefaultAzureCredential()
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

        # OpenAI embedding client
        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding,
        )

        # Database setup
        self.db = PostgreSQLManager(max_db_connections)
        self.stats = AzureSearchProcessingStats()

        # embedding dimension for text-embedding-ada-002
        self.embedding_dim = 1536

        # max docs per Azure Search index call (service limit)
        self.max_search_chunk_size = 1000

    # ------------------------------------------------------------------
    # Helpers to build embedding inputs
    # ------------------------------------------------------------------
    def _build_name_text(self, student: Dict[str, Any]) -> str:
        """Build the text used for name embedding."""
        first_name = (student.get("legalFirstName") or "").strip()
        last_name = (student.get("legalLastName") or "").strip()
        middle_names = (student.get("legalMiddleNames") or "").strip()

        if first_name == "NULL":
            first_name = ""
        if last_name == "NULL":
            last_name = ""
        if middle_names == "NULL":
            middle_names = ""

        parts = [p for p in [first_name, middle_names, last_name] if p]
        if not parts:
            return ""

        return " ".join(parts) + "."

    def generate_embeddings_for_batch(
        self,
        students: List[Dict[str, Any]],
        max_inputs_per_call: int = 16,
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of students using text-embedding-ada-002.
        Uses batched calls (up to max_inputs_per_call per request).
        """
        dim = self.embedding_dim
        zero_vec = [0.0] * dim

        # Build texts
        texts: List[Optional[str]] = []
        for s in students:
            txt = self._build_name_text(s)
            texts.append(txt if txt else None)

        embeddings: List[Optional[List[float]]] = [None] * len(students)

        indexed_texts = [(i, t) for i, t in enumerate(texts) if t]

        for start in range(0, len(indexed_texts), max_inputs_per_call):
            chunk = indexed_texts[start : start + max_inputs_per_call]
            idxs = [i for (i, _) in chunk]
            inputs = [t for (_, t) in chunk]

            try:
                resp = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=inputs,
                )
                for j, idx in enumerate(idxs):
                    embeddings[idx] = resp.data[j].embedding
            except Exception as e:
                print(f"[WARN] Embedding batch failed ({len(inputs)} inputs): {e}")
                for idx in idxs:
                    embeddings[idx] = zero_vec

        # Fill any missing embeddings (e.g., empty names) with zero vector
        return [emb if emb is not None else zero_vec for emb in embeddings]

    # ------------------------------------------------------------------
    # Document preparation
    # ------------------------------------------------------------------
    def _prepare_search_document(
        self,
        student: Dict[str, Any],
        embedding: List[float],
    ) -> Dict[str, Any]:
        """Prepare student data for Azure Search index."""
        return {
            "student_id": str(student["student_id"]),
            "pen": student.get("pen") if student.get("pen") != "NULL" else None,
            "legalFirstName": student.get("legalFirstName")
            if student.get("legalFirstName") != "NULL"
            else None,
            "legalMiddleNames": student.get("legalMiddleNames")
            if student.get("legalMiddleNames") != "NULL"
            else None,
            "legalLastName": student.get("legalLastName")
            if student.get("legalLastName") != "NULL"
            else None,
            "nameEmbedding": embedding,
            "dob": student.get("dob") if student.get("dob") != "NULL" else None,
            "sexCode": student.get("sexCode")
            if student.get("sexCode") != "NULL"
            else None,
            "postalCode": student.get("postalCode")
            if student.get("postalCode") != "NULL"
            else None,
            "mincode": student.get("mincode") if student.get("mincode") != "NULL" else None,
            "gradeCode": student.get("gradeCode")
            if student.get("gradeCode") != "NULL"
            else None,
            "localID": student.get("localID") if student.get("localID") != "NULL" else None,
        }

    def _row_to_student(self, row: Any) -> Dict[str, Any]:
        """Convert a DB row to a student dict (before embedding)."""
        return {
            "student_id": row["student_id"],  # keep original type for keyset pagination
            "pen": row["pen"],
            "legalFirstName": row["legal_first_name"],
            "legalLastName": row["legal_last_name"],
            "legalMiddleNames": row["legal_middle_names"],
            "dob": row["dob"],
            "sexCode": row["sex_code"],
            "postalCode": row["postal_code"],
            "mincode": row["mincode"],
            "gradeCode": row["grade_code"],
            "localID": row["local_id"],
        }

    # ------------------------------------------------------------------
    # Azure Search upload (chunked)
    # ------------------------------------------------------------------
    async def _batch_upload(
        self,
        documents: List[Dict[str, Any]],
        max_chunk_size: int = None,
    ) -> int:
        """
        Upload documents to Azure Search in safe chunks.

        - Azure AI Search supports up to ~1000 docs or 16 MB per request.
        - We split the input list into chunks to avoid 413 errors and SDK bugs.
        """
        if not documents:
            return 0

        if max_chunk_size is None:
            max_chunk_size = self.max_search_chunk_size

        total_uploaded = 0
        loop = asyncio.get_event_loop()

        for start in range(0, len(documents), max_chunk_size):
            chunk = documents[start : start + max_chunk_size]
            try:
                upload_result = await loop.run_in_executor(
                    None,
                    lambda c=chunk: self.search_client.upload_documents(documents=c),
                )

                for result in upload_result:
                    if result.succeeded:
                        total_uploaded += 1

            except HttpResponseError as e:
                print(
                    f"[ERROR] Azure Search upload failed for chunk of {len(chunk)} docs: {e}"
                )
            except Exception as e:
                print(
                    f"[ERROR] Unexpected Azure Search upload error for chunk of {len(chunk)} docs: {e}"
                )

        return total_uploaded

    # ------------------------------------------------------------------
    # Single batch by offset (for testing / manual runs)
    # ------------------------------------------------------------------
    async def import_one_batch(self, offset: int = 0, batch_size: int = 100) -> int:
        """
        Import a single batch using LIMIT/OFFSET.

        - batch_size is how many rows we pull from Postgres (can be 1000, 5000, 10000…)
        - Azure Search uploads are still chunked to 1000 docs inside _batch_upload.
        """
        start_time = time.time()

        await self.db.create_pool()
        try:
            async with self.db.connection_pool.acquire() as conn:
                query = """
                    SELECT student_id,
                           COALESCE(pen, 'NULL') AS pen,
                           COALESCE(legal_first_name, 'NULL') AS legal_first_name,
                           COALESCE(legal_last_name, 'NULL') AS legal_last_name,
                           COALESCE(legal_middle_names, 'NULL') AS legal_middle_names,
                           COALESCE(dob::text, 'NULL') AS dob,
                           COALESCE(sex_code, 'NULL') AS sex_code,
                           COALESCE(postal_code, 'NULL') AS postal_code,
                           COALESCE(mincode, 'NULL') AS mincode,
                           COALESCE(grade_code, 'NULL') AS grade_code,
                           COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') AS local_id
                    FROM "api_pen_match_v2".student
                    ORDER BY student_id ASC
                    LIMIT $1 OFFSET $2
                """
                rows = await conn.fetch(query, batch_size, offset)
                if not rows:
                    print("No rows returned for this batch.")
                    return 0

            students: List[Dict[str, Any]] = [self._row_to_student(row) for row in rows]

            embeddings = self.generate_embeddings_for_batch(students)

            documents = [
                self._prepare_search_document(
                    {**student, "student_id": str(student["student_id"])},
                    emb,
                )
                for student, emb in zip(students, embeddings)
            ]

            uploaded = await self._batch_upload(documents)
            elapsed = time.time() - start_time
            print(f"Batch import completed: {uploaded}/{len(rows)} in {elapsed:.1f}s")
            return uploaded

        finally:
            await self.db.close()

    # ------------------------------------------------------------------
    # Import all students (optimized for 4M+ using keyset pagination)
    # ------------------------------------------------------------------
    async def import_all_students(self, batch_size: int = 1000) -> int:
        """
        Import all students in batches using keyset pagination on student_id.

        - batch_size controls how many rows per DB query (1000, 5000, 10000, etc.).
        - Each batch is still uploaded to Search in 1000-doc chunks.
        """
        start_time = time.time()
        self.stats = AzureSearchProcessingStats(start_time=start_time)

        await self.db.create_pool()
        try:
            # Get total count once
            async with self.db.connection_pool.acquire() as conn:
                count_result = await conn.fetchval(
                    'SELECT COUNT(*) FROM "api_pen_match_v2".student'
                )
                total_count = int(count_result or 0)

            print(f"Starting import of {total_count:,} students...")

            last_student_id = None

            while True:
                async with self.db.connection_pool.acquire() as conn:
                    if last_student_id is None:
                        query = """
                            SELECT student_id,
                                   COALESCE(pen, 'NULL') AS pen,
                                   COALESCE(legal_first_name, 'NULL') AS legal_first_name,
                                   COALESCE(legal_last_name, 'NULL') AS legal_last_name,
                                   COALESCE(legal_middle_names, 'NULL') AS legal_middle_names,
                                   COALESCE(dob::text, 'NULL') AS dob,
                                   COALESCE(sex_code, 'NULL') AS sex_code,
                                   COALESCE(postal_code, 'NULL') AS postal_code,
                                   COALESCE(mincode, 'NULL') AS mincode,
                                   COALESCE(grade_code, 'NULL') AS grade_code,
                                   COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') AS local_id
                            FROM "api_pen_match_v2".student
                            ORDER BY student_id ASC
                            LIMIT $1
                        """
                        rows = await conn.fetch(query, batch_size)
                    else:
                        query = """
                            SELECT student_id,
                                   COALESCE(pen, 'NULL') AS pen,
                                   COALESCE(legal_first_name, 'NULL') AS legal_first_name,
                                   COALESCE(legal_last_name, 'NULL') AS legal_last_name,
                                   COALESCE(legal_middle_names, 'NULL') AS legal_middle_names,
                                   COALESCE(dob::text, 'NULL') AS dob,
                                   COALESCE(sex_code, 'NULL') AS sex_code,
                                   COALESCE(postal_code, 'NULL') AS postal_code,
                                   COALESCE(mincode, 'NULL') AS mincode,
                                   COALESCE(grade_code, 'NULL') AS grade_code,
                                   COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') AS local_id
                            FROM "api_pen_match_v2".student
                            WHERE student_id > $2
                            ORDER BY student_id ASC
                            LIMIT $1
                        """
                        rows = await conn.fetch(query, batch_size, last_student_id)

                if not rows:
                    break

                last_student_id = rows[-1]["student_id"]
                students: List[Dict[str, Any]] = [self._row_to_student(row) for row in rows]

                embeddings = self.generate_embeddings_for_batch(students)

                documents = [
                    self._prepare_search_document(
                        {**student, "student_id": str(student["student_id"])}, emb
                    )
                    for student, emb in zip(students, embeddings)
                ]

                uploaded = await self._batch_upload(documents)

                self.stats.total_processed += uploaded
                self.stats.total_failed += len(rows) - uploaded
                self.stats.batches_completed += 1

                # Light progress logging
                if (
                    self.stats.total_processed % 10000 == 0
                    or self.stats.total_processed == total_count
                ):
                    elapsed = time.time() - start_time
                    rate = self.stats.total_processed / max(elapsed, 1)
                    print(
                        f"{self.stats.total_processed}/{total_count} "
                        f"({rate:.0f} docs/sec, batches={self.stats.batches_completed})"
                    )

            total_time = time.time() - start_time
            rate = self.stats.total_processed / max(total_time, 1)
            print(
                f"Import completed: {self.stats.total_processed:,} students "
                f"in {total_time:.1f}s ({rate:.0f}/sec). "
                f"Failed: {self.stats.total_failed:,}"
            )
            return self.stats.total_processed

        finally:
            await self.db.close()

    # ------------------------------------------------------------------
    # Import by specific names (can be many names, many students)
    # ------------------------------------------------------------------
    async def import_all_records_by_names(self, target_names: List[tuple]) -> int:
        """
        Import all student records that match specified name pairs.

        - target_names can have 10, 100, 10,000+ name pairs.
        - For each name pair, we load all matching students, embed them in batch,
          and upload them in 1000-doc chunks.
        """
        start_time = time.time()

        await self.db.create_pool()
        try:
            total_uploaded = 0

            for i, (first_name, last_name) in enumerate(target_names, 1):
                async with self.db.connection_pool.acquire() as conn:
                    query = """
                        SELECT student_id,
                               COALESCE(pen, 'NULL') AS pen,
                               COALESCE(legal_first_name, 'NULL') AS legal_first_name,
                               COALESCE(legal_last_name, 'NULL') AS legal_last_name,
                               COALESCE(legal_middle_names, 'NULL') AS legal_middle_names,
                               COALESCE(dob::text, 'NULL') AS dob,
                               COALESCE(sex_code, 'NULL') AS sex_code,
                               COALESCE(postal_code, 'NULL') AS postal_code,
                               COALESCE(mincode, 'NULL') AS mincode,
                               COALESCE(grade_code, 'NULL') AS grade_code,
                               COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') AS local_id
                        FROM "api_pen_match_v2".student
                        WHERE LOWER(TRIM(legal_first_name)) = LOWER($1)
                          AND LOWER(TRIM(legal_last_name)) = LOWER($2)
                        ORDER BY student_id ASC
                    """
                    rows = await conn.fetch(query, first_name.strip(), last_name.strip())

                if not rows:
                    continue

                students: List[Dict[str, Any]] = [self._row_to_student(row) for row in rows]
                embeddings = self.generate_embeddings_for_batch(students)

                documents = [
                    self._prepare_search_document(
                        {**student, "student_id": str(student["student_id"])}, emb
                    )
                    for student, emb in zip(students, embeddings)
                ]

                uploaded_for_name = await self._batch_upload(documents)
                total_uploaded += uploaded_for_name

                print(
                    f"[{i}/{len(target_names)}] {first_name} {last_name}: "
                    f"{uploaded_for_name}/{len(rows)} uploaded"
                )

            total_time = time.time() - start_time
            print(f"Name import completed: {total_uploaded} students in {total_time:.1f}s")
            return total_uploaded

        finally:
            await self.db.close()


# ----------------------------------------------------------------------
# Standalone functions for compatibility
# ----------------------------------------------------------------------
async def import_student_data():
    """Import sample student data for testing (1 row)."""
    service = AzureSearchImportService()
    return await service.import_one_batch(0, 1)


def import_one_batch(offset: int = 0, batch_size: int = 100) -> int:
    """Synchronous wrapper for batch import by offset (testing/manual)."""
    service = AzureSearchImportService()
    return asyncio.run(service.import_one_batch(offset, batch_size))


def import_all_students(batch_size: int = 1000) -> int:
    """Synchronous wrapper for importing all students."""
    service = AzureSearchImportService()
    return asyncio.run(service.import_all_students(batch_size))


def import_by_names(target_names: List[tuple]) -> int:
    """Synchronous wrapper for importing by names."""
    service = AzureSearchImportService()
    return asyncio.run(service.import_all_records_by_names(target_names))


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    async def main() -> int:
        start_time = time.time()
        service = AzureSearchImportService()

        try:
            if len(sys.argv) > 1:
                if sys.argv[1] == "all":
                    # e.g. python -m azure_search.azure_search_import all 10000
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
                    result = await service.import_all_students(batch_size)
                elif sys.argv[1] == "names":
                    target_names = [
                        ("ROBYN", "ANDERSON"),
                        ("MICHAEL", "LEE"),
                        ("MICHAEL", "KIM"),
                        ("DAVID", "LEE"),
                        ("MICHAEL", "WANG"),
                        ("JENNIFER", "LEE"),
                        ("MICHAEL", "LI"),
                        ("ROBERT", "LEE"),
                        ("DAVID", "WANG"),
                        ("MICHAEL", "CHEN"),
                        ("JAMES", "LEE"),
                    ]
                    result = await service.import_all_records_by_names(target_names)
                else:
                    # e.g. python -m azure_search.azure_search_import 10000 10000
                    offset = int(sys.argv[1])
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                    result = await service.import_one_batch(offset, batch_size)
            else:
                result = await service.import_one_batch()

            total_time = time.time() - start_time
            print(f"FINAL: {result} students processed in {total_time:.1f}s")
            return 0

        except Exception as e:
            total_time = time.time() - start_time
            print(f"Import failed after {total_time:.1f}s: {e}")
            return 1

    exit_code = asyncio.run(main())
    exit(exit_code)
