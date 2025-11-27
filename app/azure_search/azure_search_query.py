import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from config.settings import settings


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------

@dataclass
class SearchResult:
    student_id: str
    pen: Optional[str]
    legal_first_name: Optional[str]
    legal_middle_names: Optional[str]
    legal_last_name: Optional[str]
    dob: Optional[str]
    sex_code: Optional[str]
    postal_code: Optional[str]
    mincode: Optional[str]
    grade: Optional[str]
    local_id: Optional[str]
    search_score: float
    match_type: str = "candidate"  # "perfect" or "candidate"


@dataclass
class SearchResponse:
    perfect_matches: List[SearchResult]
    candidates: List[SearchResult]
    query: Dict[str, Any]
    total_results: int
    search_method: str
    search_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    azure_search_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    truncated_at_100: bool = False


# -------------------------------------------------------------------
# AzureSearchQuery – new, simplified logic
# -------------------------------------------------------------------

class AzureSearchQuery:
    """
    New Azure Search wrapper.

    Core ideas:
    - Use Azure AI Search built-in ranking (text + vector + scoring profile).
    - Only classify into perfect_matches vs candidates on top of that.
    - Do *minimal* hard filtering (DOB equality where safe).
    """

    def __init__(self):
        # You can move these into settings if you prefer
        self.search_endpoint = settings.azure_search_endpoint
        self.index_name = settings.azure_search_index_name
        self.scoring_profile = "student-hybrid-v1"  # configure in your index, or set to None

        self.credential = DefaultAzureCredential()
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding_3,
        )

        self._setup_logging()
        print("AzureSearchQuery (new) initialized")

    # -------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------

    def _setup_logging(self):
        log_dir = os.path.join(os.path.dirname(__file__), "..", "log")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"azure_search_debug_{timestamp}.log")

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Azure Search Debug Log - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def _log_debug(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    # -------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------

    async def search_students(self, query: Dict[str, Any]) -> SearchResponse:
        """
        Main entry:
        - If PEN or student_id present: direct lookup
        - Else: hybrid name search (text + vector)
        """
        start = time.perf_counter()
        self._log_debug("=== SEARCH STARTED ===")
        self._log_debug("Query:\n" + json.dumps(query, indent=2))

        try:
            if query.get("pen") or query.get("student_id"):
                resp = await self._direct_lookup(query)
            else:
                resp = await self._hybrid_name_search(query)

            resp.search_time_ms = (time.perf_counter() - start) * 1000.0

            self._log_debug("=== SEARCH COMPLETED ===")
            self._log_debug(
                f"Method: {resp.search_method}, "
                f"Total: {resp.total_results}, "
                f"Perfect: {len(resp.perfect_matches)}, "
                f"Candidates: {len(resp.candidates)}"
            )
            return resp

        except Exception as e:
            self._log_debug(f"SEARCH ERROR: {e}")
            return SearchResponse(
                perfect_matches=[],
                candidates=[],
                query=query,
                total_results=0,
                search_method="error",
            )

    # -------------------------------------------------------------------
    # Direct lookup – PEN / student_id
    # -------------------------------------------------------------------

    async def _direct_lookup(self, query: Dict[str, Any]) -> SearchResponse:
        self._log_debug("Using direct lookup (PEN / student_id)")
        filter_expr = None

        if query.get("pen"):
            filter_expr = f"pen eq '{query['pen']}'"
        elif query.get("student_id"):
            filter_expr = f"student_id eq '{query['student_id']}'"

        if not filter_expr:
            return SearchResponse(
                perfect_matches=[],
                candidates=[],
                query=query,
                total_results=0,
                search_method="direct_lookup_invalid",
            )

        azure_start = time.perf_counter()
        results = self.search_client.search(
            search_text="*",
            filter=filter_expr,
            top=5,
            select=[
                "student_id",
                "pen",
                "legalFirstName",
                "legalMiddleNames",
                "legalLastName",
                "dob",
                "sexCode",
                "postalCode",
                "mincode",
                "grade",
                "localID",
            ],
            include_total_count=True,
        )
        azure_ms = (time.perf_counter() - azure_start) * 1000.0

        docs = list(results)
        total_count = results.get_count() or len(docs)

        perfect = [self._map_search_result(d, "perfect") for d in docs]

        return SearchResponse(
            perfect_matches=perfect,
            candidates=[],
            query=query,
            total_results=total_count,
            search_method="direct_lookup",
            azure_search_time_ms=azure_ms,
            embedding_time_ms=0.0,
            processing_time_ms=0.0,
            truncated_at_100=False,
        )

    # -------------------------------------------------------------------
    # Hybrid name search – text + vector, rely on Azure scoring
    # -------------------------------------------------------------------

    async def _hybrid_name_search(self, query: Dict[str, Any]) -> SearchResponse:
        # Require at least first + last name
        if not query.get("legalFirstName") or not query.get("legalLastName"):
            self._log_debug("Missing required name fields")
            return SearchResponse(
                perfect_matches=[],
                candidates=[],
                query=query,
                total_results=0,
                search_method="missing_required_fields",
            )

        # 1) Embedding for the name
        emb_start = time.perf_counter()
        embedding = self._generate_name_embedding(
            query.get("legalFirstName", ""),
            query.get("legalLastName", ""),
            query.get("legalMiddleNames", ""),
        )
        emb_ms = (time.perf_counter() - emb_start) * 1000.0

        # 2) Keyword text (for full-text scoring)
        keyword_text = self._build_keyword_text(query)

        # 3) Optional filter – DOB only (high-confidence exact match)
        filter_expr = self._build_filter(query)

        # 4) Azure AI Search: text + vector, use scoring profile
        azure_start = time.perf_counter()
        vector_query = VectorizedQuery(
            vector=embedding,
            fields="nameEmbedding",
            k_nearest_neighbors=100,
        )

        results = self.search_client.search(
            search_text=keyword_text,
            vector_queries=[vector_query],
            top=100,
            search_fields=["legalFirstName", "legalMiddleNames", "legalLastName"],
            filter=filter_expr,
            scoring_profile=self.scoring_profile,
            select=[
                "student_id",
                "pen",
                "legalFirstName",
                "legalMiddleNames",
                "legalLastName",
                "dob",
                "sexCode",
                "postalCode",
                "mincode",
                "grade",
                "localID",
            ],
            include_total_count=True,
        )
        azure_ms = (time.perf_counter() - azure_start) * 1000.0

        docs = list(results)
        total_count = results.get_count() or len(docs)

        proc_start = time.perf_counter()

        # 5) Classify using simple rules on top of Azure's ranking
        mapped: List[SearchResult] = []
        for d in docs:
            mapped.append(
                self._map_search_result(
                    d,
                    match_type="candidate",  # temporary, will be overwritten below
                )
            )

        perfect_matches: List[SearchResult] = []
        candidates: List[SearchResult] = []

        # 5.1 First: exact "perfect" matches (all provided fields match exactly)
        for r in mapped:
            if self._is_perfect_match(query, r):
                r.match_type = "perfect"
                perfect_matches.append(r)

        if perfect_matches:
            # Keep all other returned docs as candidates (already ranked by Azure)
            ids_perfect = {r.student_id for r in perfect_matches}
            for r in mapped:
                if r.student_id not in ids_perfect:
                    r.match_type = "candidate"
                    candidates.append(r)
            search_method = (
                "single_perfect_match"
                if len(perfect_matches) == 1
                else "multiple_perfect_matches"
            )
        else:
            # 5.2 No full perfect match – use 2+ exact non-name fields as a strong hint
            two_field_exact: List[SearchResult] = []
            for r in mapped:
                if self._count_exact_support_fields(query, r) >= 2:
                    two_field_exact.append(r)

            if len(two_field_exact) == 1:
                # Treat that one as "perfect" (strong match by DOB/postal/mincode/etc.)
                strong = two_field_exact[0]
                strong.match_type = "perfect"
                perfect_matches = [strong]
                candidates = [r for r in mapped if r.student_id != strong.student_id]
                search_method = "single_two_field_exact"
            else:
                # Either multiple strongs or none – just treat all as candidates
                perfect_matches = []
                candidates = mapped
                search_method = "candidates_only"

        # 6) Truncation info (we already used top=100)
        truncated_at_100 = total_count > len(mapped)
        proc_ms = (time.perf_counter() - proc_start) * 1000.0

        return SearchResponse(
            perfect_matches=perfect_matches,
            candidates=candidates,
            query=query,
            total_results=total_count,
            search_method=search_method,
            embedding_time_ms=emb_ms,
            azure_search_time_ms=azure_ms,
            processing_time_ms=proc_ms,
            truncated_at_100=truncated_at_100,
        )

    # -------------------------------------------------------------------
    # Helpers – embedding, text, filter, mapping, matching
    # -------------------------------------------------------------------

    def _generate_name_embedding(
        self, first_name: str, last_name: str, middle_names: str = ""
    ) -> List[float]:
        parts = []
        if first_name:
            parts.append(first_name.strip())
        if middle_names:
            parts.append(middle_names.strip())
        if last_name:
            parts.append(last_name.strip())
        text = " ".join(parts) + "."

        self._log_debug(f"Embedding text: '{text}'")

        resp = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        emb = resp.data[0].embedding
        self._log_debug(f"Embedding generated, dim={len(emb)}")
        return emb

    def _build_keyword_text(self, query: Dict[str, Any]) -> str:
        """
        Simple concatenation for full-text search.
        Name is dominant; other fields just give extra hints.
        """
        parts: List[str] = []

        fn = query.get("legalFirstName")
        mn = query.get("legalMiddleNames")
        ln = query.get("legalLastName")
        if fn:
            parts.append(str(fn))
        if mn:
            parts.append(str(mn))
        if ln:
            parts.append(str(ln))

        if query.get("dob"):
            parts.append(str(query["dob"]))
        if query.get("postalCode"):
            parts.append(str(query["postalCode"]))
        if query.get("mincode"):
            parts.append(str(query["mincode"]))
        if query.get("sexCode"):
            parts.append(str(query["sexCode"]))

        text = " ".join(parts)
        self._log_debug(f"Keyword search_text: '{text}'")
        return text

    def _build_filter(self, query: Dict[str, Any]) -> Optional[str]:
        """
        Only use DOB as a strict filter (high confidence).
        Other fields are left to scoring (weights in scoring profile).
        """
        dob = query.get("dob")
        if dob:
            filt = f"dob eq '{dob}'"
            self._log_debug(f"Filter expression: {filt}")
            return filt
        self._log_debug("No filter expression")
        return None

    def _map_search_result(self, doc: Dict[str, Any], match_type: str) -> SearchResult:
        score = float(doc.get("@search.score", 0.0))
        return SearchResult(
            student_id=doc.get("student_id", "") or "",
            pen=doc.get("pen"),
            legal_first_name=doc.get("legalFirstName"),
            legal_middle_names=doc.get("legalMiddleNames"),
            legal_last_name=doc.get("legalLastName"),
            dob=doc.get("dob"),
            sex_code=doc.get("sexCode"),
            postal_code=doc.get("postalCode"),
            mincode=doc.get("mincode"),
            grade=doc.get("grade"),
            local_id=doc.get("localID"),
            search_score=score,
            match_type=match_type,
        )

    def _normalize(self, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip().upper()

    def _is_perfect_match(self, query: Dict[str, Any], r: SearchResult) -> bool:
        """
        All provided fields in query must match exactly.
        """
        field_map = {
            "pen": "pen",
            "legalFirstName": "legal_first_name",
            "legalMiddleNames": "legal_middle_names",
            "legalLastName": "legal_last_name",
            "dob": "dob",
            "sexCode": "sex_code",
            "postalCode": "postal_code",
            "mincode": "mincode",
            "grade": "grade",
            "localID": "local_id",
        }

        for q_field, r_attr in field_map.items():
            q_val = query.get(q_field)
            if not q_val or q_val == "NULL":
                continue
            r_val = getattr(r, r_attr, None)
            if not r_val or r_val == "NULL":
                return False
            if self._normalize(q_val) != self._normalize(r_val):
                return False
        return True

    def _count_exact_support_fields(self, query: Dict[str, Any], r: SearchResult) -> int:
        """
        Count exact matches among non-name support fields:
        dob, sexCode, postalCode, mincode, grade, localID.
        """
        fields = [
            ("dob", "dob"),
            ("sexCode", "sex_code"),
            ("postalCode", "postal_code"),
            ("mincode", "mincode"),
            ("grade", "grade"),
            ("localID", "local_id"),
        ]
        count = 0
        for q_field, r_attr in fields:
            q_val = query.get(q_field)
            if not q_val or q_val == "NULL":
                continue
            r_val = getattr(r, r_attr, None)
            if not r_val or r_val == "NULL":
                continue
            if self._normalize(q_val) == self._normalize(r_val):
                count += 1
        return count

    # -------------------------------------------------------------------
    # Debug printing
    # -------------------------------------------------------------------

    def print_search_response(self, resp: SearchResponse, debug_limit: int = 5):
        print("\n=== SEARCH RESULTS ===")
        print(f"Search Method: {resp.search_method}")
        print(f"Total Results (from Azure): {resp.total_results}")
        print(f"Perfect Matches: {len(resp.perfect_matches)}")
        print(f"Candidates (returned): {len(resp.candidates)}")
        print(f"Truncated at 100: {resp.truncated_at_100}")
        print("TIMING (ms):")
        print(f"  Total:      {resp.search_time_ms:.2f}")
        print(f"  Embedding:  {resp.embedding_time_ms:.2f}")
        print(f"  Azure:      {resp.azure_search_time_ms:.2f}")
        print(f"  Processing: {resp.processing_time_ms:.2f}")
        print(f"Debug Log: {self.log_file}")

        if resp.perfect_matches:
            print(f"\n--- PERFECT MATCHES (up to {debug_limit}) ---")
            for i, r in enumerate(resp.perfect_matches[:debug_limit], start=1):
                self._print_result(i, r)
            if len(resp.perfect_matches) > debug_limit:
                print(f"... and {len(resp.perfect_matches) - debug_limit} more")

        if resp.candidates:
            print(f"\n--- CANDIDATES (up to {debug_limit}) ---")
            for i, r in enumerate(resp.candidates[:debug_limit], start=1):
                self._print_result(i, r)
            if len(resp.candidates) > debug_limit:
                print(f"... and {len(resp.candidates) - debug_limit} more")

    def _print_result(self, idx: int, r: SearchResult):
        name = " ".join(
            x for x in [r.legal_first_name, r.legal_middle_names, r.legal_last_name] if x
        )
        print(f"\n{idx}. {name}")
        print(f"   Student ID: {r.student_id} | PEN: {r.pen}")
        print(f"   DOB: {r.dob} | Sex: {r.sex_code}")
        print(f"   Postal: {r.postal_code} | Mincode: {r.mincode}")
        print(f"   Grade: {r.grade} | LocalID: {r.local_id}")
        print(f"   Score: {r.search_score:.4f} ({r.match_type})")


# -------------------------------------------------------------------
# Optional: small test harness
# -------------------------------------------------------------------

async def run_test_suite():
    q = AzureSearchQuery()

    base = {"legalFirstName": "MICHAEL", "legalLastName": "LEE"}

    tests = [
        ("Name Only", base),
        ("Name + Middle", {**base, "legalMiddleNames": "RICHARD"}),
        ("Name + DOB", {**base, "dob": "2001-02-10"}),
        ("Name + All", {
            **base,
            "legalMiddleNames": "RICHARD",
            "dob": "2001-02-10",
            "sexCode": "M",
            "postalCode": "V3N1H4",
            "mincode": "05757079",
        }),
        ("Direct PEN", {"pen": "124809765"}),
    ]

    print("Running Azure Search Test Suite (new logic)")
    print("Debug logs in app/log/")

    for name, query in tests:
        print("\n" + "=" * 60)
        print(f"TEST: {name}")
        print(f"Query: {query}")
        print("=" * 60)
        resp = await q.search_students(query)
        q.print_search_response(resp, debug_limit=5)


if __name__ == "__main__":
    asyncio.run(run_test_suite())
