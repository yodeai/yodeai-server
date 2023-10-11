if git diff --quiet HEAD@{1} HEAD requirements.txt; then
    echo "No changes in requirements.txt. No action required."
else
    echo "Updating Python dependencies..."
    pip install -r requirements.txt
fi