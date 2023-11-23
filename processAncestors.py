from DB import supabaseClient

def processAncestors(block_id, lens_id, remove):
    print("in process ancestors now")
    data, data_count= supabaseClient.from_('lens').select('parent_id').eq('lens_id', lens_id).execute()
    lens_id = data[1][0].get("parent_id")

    while lens_id != -1:
        # Check if the mapping already exists
        data, data_count = supabaseClient.from_('lens_blocks').select('*').eq('lens_id', lens_id).eq('block_id', block_id).execute()
        existing_mapping = data[1][0] if len(data[1]) != 0 else None

        if existing_mapping:
            count = existing_mapping.get('count', 0)

            if remove:
                # If removing, decrement count
                count -= 1
            else:
                # If adding, increment count
                count += 1

            if count > 0:
                # Update count in the lens_blocks table
                supabaseClient.from_('lens_blocks').update({'count': count}).eq('lens_id', lens_id).eq('block_id', block_id).execute()
            else:
                # If count is zero, remove the entry from the lens_blocks table
                supabaseClient.from_('lens_blocks').delete().eq('lens_id', lens_id).eq('block_id', block_id).execute()

        elif not remove:
            # If mapping does not exist, add it
            supabaseClient.from_('lens_blocks').insert([
                {'lens_id': lens_id, 'block_id': block_id, 'count': 1, "direct_child": False}
            ]).execute()

        # Get parent lens_id
        data, data_count= supabaseClient.from_('lens').select('parent_id').eq('lens_id', lens_id).execute()
        lens_id = data[1][0].get("parent_id")