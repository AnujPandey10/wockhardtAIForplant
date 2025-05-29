import os
from sentence_transformers import SentenceTransformer
import shutil

def download_model():
    # Model configuration
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    local_model_path = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(local_model_path, exist_ok=True)
    
    print(f"Downloading model to {local_model_path}...")
    try:
        # Download and save the model
        model = SentenceTransformer(model_name)
        model.save(local_model_path)
        print(f"Model successfully downloaded and saved to {local_model_path}")
        
        # List saved files
        print("\nVerifying saved files:")
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                print(f"- {os.path.join(root, file)}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("\nIf you're behind a proxy, try setting these environment variables:")
        print("set HTTPS_PROXY=http://username:password@proxy:port")
        print("set HTTP_PROXY=http://username:password@proxy:port")

if __name__ == "__main__":
    download_model()
