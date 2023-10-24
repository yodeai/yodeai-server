import os
from supabase import create_client
from dotenv import load_dotenv

from debug.tools import clearConsole
load_dotenv(dotenv_path='.env.local')


def getSupabaseClient():
    url: str = os.environ.get("PUBLIC_SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url:
        raise Exception('SUPABASE_URL environment variable is not defined')
    if not key:
        raise Exception('supabasekey environment variable is not defined')
    return create_client(url, key)

supabaseClient=getSupabaseClient()

def getBlockIDsOfLens(lensID):
    data, error = supabaseClient.from_('lens_blocks').select('block_id').eq('lens_id', lensID).execute()
    if error[1]:
        print(f"Error in getBlockIDsOfLens/retrieving from DB: {error}")
    block_ids = [item['block_id'] for item in data[1]]
    return block_ids

def getBlockTitles(blockIDs):
    data, error = supabaseClient.from_('block').select('block_id', 'title').in_('block_id', blockIDs).execute()
    if error[1]:
        print(f"Error in getBlockIDsOfLens/retrieving from DB: {error}")
    return data[1]


if __name__ == "__main__":
    print(getBlockIDsOfLens(6))
