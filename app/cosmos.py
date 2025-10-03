"""
Azure Cosmos DB Test Suite
Tests connectivity, CRUD operations, and performance for the PEN Match API
"""

import os
import json
import time
import uuid
from datetime import datetime, date
from typing import List, Dict, Any
from azure.cosmos import CosmosClient, exceptions
import random

class CosmosDBTester:
    def __init__(self):
        self.endpoint = os.environ.get('AZURE_COSMOSDB_ENDPOINT')
        self.key = os.environ.get('AZURE_COSMOSDB_KEY')
        self.database_name = os.environ.get('TEST_DATABASE_NAME')
        self.container_name = os.environ.get('TEST_CONTAINER_NAME')
        self.test_data_count = int(os.environ.get('TEST_DATA_COUNT', '10'))
        
        self.client = CosmosClient(self.endpoint, self.key)
        self.database = self.client.get_database_client(self.database_name)
        self.container = self.database.get_container_client(self.container_name)
        
        self.test_results = []

    def log_result(self, test_name: str, success: bool, message: str, duration_ms: int = None):
        """Log test results"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if duration_ms:
            result['duration_ms'] = duration_ms
        
        self.test_results.append(result)
        status = '✅' if success else '❌'
        print(f"{status} {test_name}: {message}")

    def test_connectivity(self) -> bool:
        """Test basic connectivity to Cosmos DB"""
        try:
            start_time = time.time()
            
            # Test database access
            db_properties = self.database.read()
            
            # Test container access
            container_properties = self.container.read()
            
            duration = int((time.time() - start_time) * 1000)
            self.log_result(
                "Connectivity Test", 
                True, 
                f"Connected to database '{db_properties['id']}' and container '{container_properties['id']}'",
                duration
            )
            return True
            
        except Exception as e:
            self.log_result("Connectivity Test", False, f"Connection failed: {str(e)}")
            return False

    def generate_test_data(self) -> List[Dict[str, Any]]:
        """Generate test PEN records"""
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa', 'William', 'Ashley']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        middle_names = ['Alexander', 'Marie', 'James', 'Elizabeth', 'Michael', 'Anne', 'Christopher', 'Nicole', 'Daniel', 'Michelle']
        
        test_records = []
        
        for i in range(1, self.test_data_count + 1):
            record = {
                'id': str(uuid.uuid4()),
                'pen': f'PEN{str(i).zfill(6)}',  # Partition key
                'legalFirstName': first_names[i % len(first_names)],
                'legalMiddleNames': middle_names[i % len(middle_names)] if random.random() > 0.3 else None,
                'legalLastName': last_names[i % len(last_names)],
                'dob': date(1980 + (i % 40), (i % 12) + 1, (i % 28) + 1).isoformat(),
                'localID': f'LOCAL{str(i).zfill(4)}' if random.random() > 0.5 else None,
                'createdAt': datetime.now().isoformat(),
                'testRecord': True
            }
            test_records.append(record)
        
        return test_records

    def test_insert_data(self) -> bool:
        """Test inserting test data"""
        try:
            start_time = time.time()
            test_records = self.generate_test_data()
            
            inserted_count = 0
            for record in test_records:
                try:
                    self.container.create_item(body=record)
                    inserted_count += 1
                except Exception as e:
                    print(f"Failed to insert record {record['pen']}: {str(e)}")
            
            duration = int((time.time() - start_time) * 1000)
            self.log_result(
                "Data Insertion", 
                inserted_count > 0, 
                f"Inserted {inserted_count}/{len(test_records)} records",
                duration
            )
            return inserted_count > 0
            
        except Exception as e:
            self.log_result("Data Insertion", False, f"Insert failed: {str(e)}")
            return False

    def test_crud_operations(self) -> bool:
        """Test CRUD operations"""
        try:
            # CREATE
            test_record = {
                'id': str(uuid.uuid4()),
                'pen': 'TESTPEN001',
                'legalFirstName': 'Test',
                'legalMiddleNames': 'CRUD',
                'legalLastName': 'User',
                'dob': '1990-01-01',
                'localID': 'TESTLOCAL001',
                'createdAt': datetime.now().isoformat(),
                'testRecord': True
            }
            
            created_item = self.container.create_item(body=test_record)
            print("   CREATE: Record created successfully")
            
            # READ
            read_item = self.container.read_item(item=created_item['id'], partition_key='TESTPEN001')
            print(f"   READ: Retrieved {read_item['legalFirstName']} {read_item['legalLastName']}")
            
            # UPDATE
            read_item['legalMiddleNames'] = 'Updated'
            read_item['updatedAt'] = datetime.now().isoformat()
            updated_item = self.container.replace_item(item=read_item['id'], body=read_item)
            print("   UPDATE: Record updated successfully")
            
            # DELETE
            self.container.delete_item(item=created_item['id'], partition_key='TESTPEN001')
            print("   DELETE: Record deleted successfully")
            
            self.log_result("CRUD Operations", True, "All CRUD operations completed successfully")
            return True
            
        except Exception as e:
            self.log_result("CRUD Operations", False, f"CRUD operations failed: {str(e)}")
            return False

    def test_query_operations(self) -> bool:
        """Test various query operations"""
        try:
            queries = [
                {
                    'name': 'Find by First Name',
                    'query': "SELECT * FROM c WHERE c.legalFirstName = 'John'"
                },
                {
                    'name': 'Find by Last Name', 
                    'query': "SELECT * FROM c WHERE c.legalLastName = 'Smith'"
                },
                {
                    'name': 'Find by DOB Range',
                    'query': "SELECT * FROM c WHERE c.dob >= '1990-01-01' AND c.dob <= '1999-12-31'"
                },
                {
                    'name': 'Count Test Records',
                    'query': "SELECT VALUE COUNT(1) FROM c WHERE c.testRecord = true"
                }
            ]
            
            all_successful = True
            for query_test in queries:
                try:
                    start_time = time.time()
                    items = list(self.container.query_items(
                        query=query_test['query'],
                        enable_cross_partition_query=True
                    ))
                    duration = int((time.time() - start_time) * 1000)
                    
                    result_count = len(items) if isinstance(items, list) else items[0] if items else 0
                    print(f"   {query_test['name']}: {result_count} results in {duration}ms")
                    
                except Exception as e:
                    print(f"   {query_test['name']}: Failed - {str(e)}")
                    all_successful = False
            
            self.log_result("Query Operations", all_successful, f"Completed {len(queries)} query tests")
            return all_successful
            
        except Exception as e:
            self.log_result("Query Operations", False, f"Query operations failed: {str(e)}")
            return False

    def test_performance(self) -> bool:
        """Test performance metrics"""
        try:
            start_time = time.time()
            
            # Query all test records
            items = list(self.container.query_items(
                query="SELECT * FROM c WHERE c.testRecord = true",
                enable_cross_partition_query=True
            ))
            
            query_duration = int((time.time() - start_time) * 1000)
            
            # Get container information
            container_info = self.container.read()
            
            performance_info = {
                'records_returned': len(items),
                'query_time_ms': query_duration,
                'container_id': container_info['id'],
                'partition_key': container_info['partitionKey']['paths'][0],
                'indexing_mode': container_info['indexingPolicy']['indexingMode']
            }
            
            self.log_result(
                "Performance Test", 
                True, 
                f"Retrieved {performance_info['records_returned']} records in {performance_info['query_time_ms']}ms",
                query_duration
            )
            
            return True
            
        except Exception as e:
            self.log_result("Performance Test", False, f"Performance test failed: {str(e)}")
            return False

    def export_test_data(self) -> bool:
        """Export test data for analysis"""
        try:
            items = list(self.container.query_items(
                query="SELECT * FROM c WHERE c.testRecord = true",
                enable_cross_partition_query=True
            ))
            
            export_data = {
                'exportDate': datetime.now().isoformat(),
                'recordCount': len(items),
                'records': items
            }
            
            with open('test-data-export.json', 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.log_result("Data Export", True, f"Exported {len(items)} records to test-data-export.json")
            return True
            
        except Exception as e:
            self.log_result("Data Export", False, f"Data export failed: {str(e)}")
            return False

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print(f"🚀 Starting Cosmos DB tests with {self.test_data_count} test records...")
        print(f"Database: {self.database_name}, Container: {self.container_name}")
        print("-" * 60)
        
        tests = [
            self.test_connectivity,
            self.test_insert_data,
            self.test_crud_operations,
            self.test_query_operations,
            self.test_performance
        ]
        
        # Export data only if running manually
        if os.environ.get('GITHUB_ACTIONS') != 'true':
            tests.append(self.export_test_data)
        
        all_passed = True
        for test in tests:
            success = test()
            if not success:
                all_passed = False
        
        # Print summary
        print("-" * 60)
        passed_count = sum(1 for result in self.test_results if result['success'])
        total_count = len(self.test_results)
        
        if all_passed:
            print(f"🎉 All tests passed! ({passed_count}/{total_count})")
        else:
            print(f"❌ Some tests failed. ({passed_count}/{total_count} passed)")
        
        return all_passed

def main():
    """Main function to run tests"""
    try:
        tester = CosmosDBTester()
        success = tester.run_all_tests()
        
        if not success:
            exit(1)
            
    except Exception as e:
        print(f"❌ Test suite failed to initialize: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()