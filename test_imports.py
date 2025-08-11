#!/usr/bin/env python3
"""
Test script to debug import issues.
"""

import sys
from pathlib import Path

print("Current working directory:", Path.cwd())
print("Python path:", sys.path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
print("After adding src:", sys.path)

try:
    print("\nTrying to import config.settings...")
    from config.settings import get_settings
    print("✅ Successfully imported config.settings")
    
    settings = get_settings()
    print(f"✅ Settings loaded: {settings.neo4j_uri}")
    
except Exception as e:
    print(f"❌ Failed to import config.settings: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nTrying to import db.ingestion...")
    from db.ingestion import Neo4jIngestor
    print("✅ Successfully imported db.ingestion")
    
except Exception as e:
    print(f"❌ Failed to import db.ingestion: {e}")
    import traceback
    traceback.print_exc()
