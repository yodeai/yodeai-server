import json
from utils import supabaseClient
from processBlock import processBlock

# Define a function to insert a row into the block table
def insertRowIntoBlockTable(owner_id, block, lens_id):
    requestBody =  {
      "block_type": "note",
      "content": block["content"],
      "title":  block["author_name"],
      "owner_id": owner_id,
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
    .insert({"block_id": block_id, "lens_id": lens_id, "direct_child": True, "count": 1})\
    .execute()
    processBlock(block_id)


# Read the JSON file
def process_json_file(owner_id, file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        for block in data:
            print("block: ", block)
            insertRowIntoBlockTable(owner_id, block)

# Example usage
if __name__ == "__main__":
    json_file_path = "dana_reviews.json"
    process_json_file(json_file_path)
