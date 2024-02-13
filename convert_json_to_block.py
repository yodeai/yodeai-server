import json
from utils import supabaseClient
import requests

LENS_ID = 972
PROCESS_BLOCK_ENDPOINT = "http://127.0.0.1:8000/processBlock"

# Define a function to process a block
def processBlock(block_id):
    response = requests.post(PROCESS_BLOCK_ENDPOINT, json={"block_id": block_id, "delay": 0})
    if response.status_code != 200:
        print(f"Failed to process block {block_id}. Status code: {response.status_code}")
    else:
        print(f"Successfully processed block {block_id}")

# Define a function to insert a row into the block table
def insertRowIntoBlockTable(block):
    requestBody =  {
      "block_type": "note",
      "content": block["content"],
      "title":  block["author_name"],
      "owner_id": "0e1fbeef-df41-46a4-b7be-d27766d395b2",
      "google_user_id": 'global',
      "original_date": block["updated"]
    }
    # insert into block table
    insert_response, _ = supabaseClient.table('block')\
    .insert(requestBody)\
    .execute()
    print("insert response", insert_response)
    block_id = insert_response[1][0]["block_id"]
    print('block id', block_id)
    # insert into lens_block
    lens_block_response, lens_block_error = supabaseClient.table('lens_blocks')\
    .insert({"block_id": block_id, "lens_id": LENS_ID, "direct_child": True, "count": 1})\
    .execute()
    processBlock(block_id)

# Read the JSON file
def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        for block in data:
            print("block: ", block)
            insertRowIntoBlockTable(block)

# Example usage
if __name__ == "__main__":
    json_file_path = "dana_reviews.json"
    process_json_file(json_file_path)
