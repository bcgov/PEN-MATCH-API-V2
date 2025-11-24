import numpy as np
from openai import OpenAI, AzureOpenAI
from config.settings import settings

class StudentEmbedding:
    def __init__(self):
        # Configure OpenAI client
        if settings.openai_api_base_embedding:
            self.openai_client = AzureOpenAI(
                api_key=settings.openai_api_key,
                api_version="2023-05-15",
                azure_endpoint=settings.openai_api_base_embedding
            )
        else:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def _number_to_text(self, num):
        """Convert number to text representation"""
        num_to_word = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
            16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
            20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
            70: "seventy", 80: "eighty", 90: "ninety"
        }
        
        if num in num_to_word:
            return num_to_word[num]
        elif 21 <= num <= 99:
            tens = (num // 10) * 10
            ones = num % 10
            return f"{num_to_word[tens]} {num_to_word[ones]}"
        else:
            return str(num)

    def _year_to_text(self, year):
        """Convert full year to text (e.g., 1995 -> nineteen ninety five)"""
        if year < 1000 or year > 9999:
            return str(year)
        
        # Handle years like 1995, 2005, etc.
        if year >= 2000:
            # For 2000s: two thousand five, two thousand twenty, etc.
            if year == 2000:
                return "two thousand"
            elif year < 2010:
                return f"two thousand {self._number_to_text(year % 10)}"
            else:
                return f"two thousand {self._number_to_text(year % 100)}"
        else:
            # For 1900s: nineteen ninety five, etc.
            century = year // 100
            remainder = year % 100
            
            century_text = self._number_to_text(century)
            if remainder == 0:
                return f"{century_text} hundred"
            else:
                remainder_text = self._number_to_text(remainder)
                return f"{century_text} {remainder_text}"

    def _month_to_text(self, month_num):
        """Convert month number to text"""
        months = {
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        }
        return months.get(month_num, "")

    def _format_dob(self, dob_str):
        """Convert date string to text format: Born on September twenty, nineteen ninety five"""
        try:
            if not dob_str or dob_str == 'NULL':
                return ""
            
            # Parse date (format: YYYY-MM-DD)
            year, month, day = dob_str.split('-')
            year_num = int(year)
            month_num = int(month)
            day_num = int(day)
            
            year_text = self._year_to_text(year_num)
            month_text = self._month_to_text(month_num)
            day_text = self._number_to_text(day_num)
            
            return f"Born on {month_text} {day_text}, {year_text}"
        except:
            return ""

    def _format_sex(self, sex_code):
        """Convert sex code to full text"""
        if sex_code == 'M':
            return "male"
        elif sex_code == 'F':
            return "female"
        else:
            return ""

    def student_to_text(self, student):
        """Convert student data to text for embedding"""
        parts = []
        
        # First name
        if student.get("legalFirstName") and student["legalFirstName"] != 'NULL':
            parts.append(f"First name: {student['legalFirstName']}.")
        
        # Last name
        if student.get("legalLastName") and student["legalLastName"] != 'NULL':
            parts.append(f"Last name: {student['legalLastName']}.")
        
        # Middle name
        if student.get("legalMiddleNames") and student["legalMiddleNames"] != 'NULL':
            parts.append(f"Middle name: {student['legalMiddleNames']}.")
        
        # DOB format: Born on September twenty, nineteen ninety five
        if student.get("dob") and student["dob"] != 'NULL':
            dob_text = self._format_dob(student["dob"])
            if dob_text:
                parts.append(f"{dob_text}.")
        
        # Sex format: male/female
        if student.get("sexCode") and student["sexCode"] != 'NULL':
            sex_text = self._format_sex(student["sexCode"])
            if sex_text:
                parts.append(f"Sex: {sex_text}.")
        
        # Postal code
        if student.get("postalCode") and student["postalCode"] != 'NULL':
            parts.append(f"Postal code: {student['postalCode']}.")
        
        # Mincode
        if student.get("mincode") and student["mincode"] != 'NULL':
            parts.append(f"mincode: {student['mincode']}.")
        
        return " ".join(parts)

    def generate_embedding(self, student):
        """Generate embedding for a student"""
        text = self.student_to_text(student)
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Failed to generate embedding: {str(e)}")

    def generate_embeddings_batch(self, students):
        """Generate embeddings for multiple students"""
        embeddings = {}
        for student in students:
            # Use pen as primary index
            pen = student.get("pen")
            if pen:
                embeddings[pen] = {
                    "embedding": self.generate_embedding(student),
                    "student_data": {
                        "pen": student.get("pen"),
                        "legalFirstName": student.get("legalFirstName"),
                        "legalLastName": student.get("legalLastName"),
                        "legalMiddleNames": student.get("legalMiddleNames"),
                        "dob": student.get("dob"),
                        "sexCode": student.get("sexCode"),
                        "postalCode": student.get("postalCode"),
                        "mincode": student.get("mincode"),
                        "localID": student.get("localID")
                    }
                }
        return embeddings
    
if __name__ == "__main__":
    try:
        embedding_service = StudentEmbedding()
        print("Testing StudentEmbedding...")
        
        # Test with sample student data
        sample_students = [
            {
                "pen": "123456789",
                "legalFirstName": "John",
                "legalLastName": "Doe",
                "legalMiddleNames": "Michael",
                "dob": "1995-09-20",
                "sexCode": "M",
                "postalCode": "V5K2A1",
                "mincode": "12345678",
                "localID": "STU001"
            },
            {
                "pen": "987654321",
                "legalFirstName": "Jane",
                "legalLastName": "Smith",
                "dob": "2005-03-15",
                "sexCode": "F",
                "postalCode": "V6B1A1",
                "mincode": "87654321",
                "localID": "STU002"
            }
        ]
        
        # Test text conversion for each student
        for i, student in enumerate(sample_students, 1):
            text = embedding_service.student_to_text(student)
            print(f"Student {i} text: {text}")
        
        # Test single embedding generation
        embedding = embedding_service.generate_embedding(sample_students[0])
        print(f"Generated embedding with {len(embedding)} dimensions")
        
        # Test batch embedding generation
        embeddings = embedding_service.generate_embeddings_batch(sample_students)
        print(f"Generated embeddings for {len(embeddings)} students")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")