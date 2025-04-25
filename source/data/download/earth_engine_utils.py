import ee

def initialize_earth_engine():
    try:
        ee.Initialize()
        return True
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return False
