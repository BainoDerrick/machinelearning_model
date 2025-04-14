import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import joblib
import sklearn
import numpy as np

class TeacherGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TeacherGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class StudentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(StudentGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def load_models():
    """Load models with the new file structure including scaler"""
    try:
        # Get absolute path to models directory (one level up from app/, then into models/)
        model_dir = Path(__file__).parent.parent / "models"  # Points to MY_PROJECT/models/
        
        # Verify all files exist
        required_files = [
            "kmeans_clusterer.joblib",
            "student_gnn.pth",
            "teacher_gnn.pth",
            "scaler.joblib"
        ]
        
        missing_files = [f for f in required_files if not (model_dir / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
        
        # Load models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print(f"Scikit-learn version in utils: {sklearn.__version__}")
        
        # Load clusterer
        try:
            kmeans = joblib.load(model_dir / "kmeans_clusterer.joblib")
            print("KMeans loaded successfully, type:", type(kmeans))
        except Exception as e:
            raise RuntimeError(f"Failed to load clusterer: {str(e)}")
        
        # Load scaler (now a dict from training)
        try:
            scaler = joblib.load(model_dir / "scaler.joblib")
            print("Scaler loaded successfully, type:", type(scaler))
            if isinstance(scaler, dict):
                print("Scaler is a dictionary with keys:", list(scaler.keys()))
            else:
                print("Scaler mean:", scaler.mean_, "scale:", scaler.scale_)
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler: {str(e)}")
        
        # Initialize models
        try:
            teacher = TeacherGNN(in_channels=5, hidden_channels=64, out_channels=2)
            teacher.load_state_dict(
                torch.load(model_dir / "teacher_gnn.pth", map_location=device)
            )
            teacher.eval()
            
            student = StudentGNN(in_channels=5, hidden_channels=32, out_channels=2)
            student.load_state_dict(
                torch.load(model_dir / "student_gnn.pth", map_location=device)
            )
            student.eval()
            
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")
        
        return teacher.to(device), student.to(device), kmeans, scaler
        
    except Exception as e:
        print(f"Debug info - Current directory: {os.getcwd()}")
        print(f"Debug info - Model directory: {model_dir}")
        print(f"Debug info - Model directory contents: {os.listdir(model_dir)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")