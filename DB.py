import time
import random
import sys
import os
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.local')


def getSupabaseClient():
    url: str = os.environ.get("PUBLIC_SUPABASE_URL")
    key: str = os.environ.get("PUBLIC_SUPABASE_ANON_KEY")
    if not url:
        raise Exception('SUPABASE_URL environment variable is not defined')
    if not key:
        raise Exception('supabasekey environment variable is not defined')
    return create_client(url, key)

mySupabase=getSupabaseClient()