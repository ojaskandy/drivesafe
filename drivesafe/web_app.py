import os
import sys

# Try importing the full app first
try:
    print("Attempting to import the full app...")
    from web.app import app
    print("Full app imported successfully!")
except Exception as e:
    print(f"Error importing full app: {str(e)}")
    print("Falling back to simple app...")
    try:
        from web.simple_app import app
        print("Simple app imported successfully!")
    except Exception as e:
        print(f"Error importing simple app: {str(e)}")
        raise

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting app on port {port}...")
    app.run(host='0.0.0.0', port=port)
