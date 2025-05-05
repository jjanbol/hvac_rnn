import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, db, get_app
from streamlit_autorefresh import st_autorefresh
import altair as alt
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
import asyncio
# --- CONFIGURATION ---
# Initialize Firebase only once
try:
    get_app()
except ValueError:
    firebase_cred_dict = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_cred_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://autonomous-hvac-default-rtdb.firebaseio.com'
        })

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Fix event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- MODEL LOADING ---
# Create a function to load the MLP model
@st.cache_resource
def load_mlp_model():
    try:
        # Use the provided MLP architecture
        class MLP(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, layers, dropout_rate):
                super().__init__()
                self.sequential = torch.nn.ModuleList()
                
                # Input layer
                self.sequential.append(torch.nn.Linear(input_dim, hidden_dim))
                self.sequential.append(torch.nn.ReLU())
                self.sequential.append(torch.nn.BatchNorm1d(hidden_dim))
                
                # Hidden layers
                for i in range(layers):
                    self.sequential.append(torch.nn.Linear(hidden_dim, hidden_dim))
                    self.sequential.append(torch.nn.ReLU())
                    self.sequential.append(torch.nn.BatchNorm1d(hidden_dim))
                
                # Output layer
                self.sequential.append(torch.nn.Linear(hidden_dim, output_dim))
                self.sequential.append(torch.nn.ReLU())
                
            def initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, torch.nn.Linear):
                        # Using kaiming_normal as specified in the config
                        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        m.bias.data.fill_(0)
                        
            def forward(self, x):
                for layer in self.sequential:
                    x = layer(x)
                return x
        
        # Create model instance with the same parameters as in the provided code
        model = MLP(
            input_dim=3,  # 3 inputs: depth, RGB, and pico
            hidden_dim=8, 
            output_dim=1,
            layers=3,
            dropout_rate=0.2  # From the config
        )
        
        # Load the saved state dict
        state_dict = torch.load('best_model', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading MLP model: {e}")
        return None

# Load traditional ML models
@st.cache_resource
def load_traditional_models():
    try:
        # Load models with version compatibility check
        def safe_load_model(path):
            try:
                return joblib.load(path)
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
                return None
        
        scaler = safe_load_model('models/scaler.joblib')
        rf_model = safe_load_model('models/random_forest.joblib')
        gb_model = safe_load_model('models/gradient_boosting.joblib')
        xgb_model = safe_load_model('models/xgboost.joblib')
        ensemble_model = safe_load_model('models/ensemble.joblib')
        
        if any(model is None for model in [scaler, rf_model, gb_model, xgb_model, ensemble_model]):
            return None
            
        return {
            'scaler': scaler,
            'rf': rf_model,
            'gb': gb_model,
            'xgb': xgb_model,
            'ensemble': ensemble_model
        }
    except Exception as e:
        st.error(f"Error loading traditional models: {e}")
        return None

# --- STREAMLIT UI CONFIG ---
st.set_page_config(
    page_title="Occupancy Dashboard",
    layout="wide"
)
st.title("Real-Time Occupancy Dashboard")

# Define the configuration for the model
config = {
    'activations': 'GELU',
    'learning_rate': 0.001,
    'max_lr': 0.006,
    'pct_start': 0.1,
    'optimizers': 'AdamW',
    'scheduler': 'OneCycleLR',  # 'ReduceLROnPlateau'
    'epochs': 25,
    'batch_size': 32,
    'weight_initialization': 'kaiming_normal',  # e.g kaiming_normal, kaiming_uniform, uniform, xavier_normal or xavier_uniform
    'dropout': 0.2
}

# Auto-refresh every 1 second
auto_refresh = st_autorefresh(interval=1000, key="datarefresh")

# --- DATA LOADERS ---
def load_yolo_depth():
    ref = db.reference('/yolo-depth')
    raw = ref.get() or {}
    rows = []
    for key, entry in raw.items():
        ts = entry.get("timestamp")
        ts_dt = None
        if isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
            except ValueError:
                pass
        else:
            try:
                ts_dt = datetime.fromtimestamp(float(ts))
            except Exception:
                pass
        # Correct for the one-hour offset
        if ts_dt:
            ts_dt -= timedelta(hours=1)
        rows.append({
            'Record ID': key,
            'Timestamp': ts_dt,
            'People Count': entry.get('People count') or entry.get('count'),
            'Fake Count': entry.get('Fake count') or entry.get('fake_count')
        })
    df = pd.DataFrame(rows)
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp', ascending=False)
    return df

def load_people_counts():
    ref = db.reference('/people_counts')
    raw = ref.get() or {}
    rows = []
    for key, entry in raw.items():
        ts = entry.get("timestamp")
        ts_dt = None
        if isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
            except ValueError:
                pass
        else:
            try:
                ts_dt = datetime.fromtimestamp(float(ts))
            except Exception:
                pass
        dets = entry.get('detections', {})
        rows.append({
            'Record ID': key,
            'Timestamp': ts_dt,
            'Count': entry.get('count'),
            'Average Confidence': entry.get('average_confidence'),
            'Detections': len(dets)
        })
    df = pd.DataFrame(rows)
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp', ascending=False)
    return df

def load_pico_count():
    ref = db.reference('/people_inside')
    raw = ref.get() or {}
    rows = []
    for key, entry in raw.items():
        ts = entry.get("timestamp")
        ts_dt = None
        if isinstance(ts, str):
            try:
                ts_dt = datetime.fromisoformat(ts)
            except ValueError:
                pass
        else:
            try:
                ts_dt = datetime.fromtimestamp(float(ts))
            except Exception:
                pass
        rows.append({
            'Record ID': key,
            'Timestamp': ts_dt,
            'Count': entry.get('count')
        })
    df = pd.DataFrame(rows)
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp', ascending=False)
    return df

# --- MODEL PREDICTION FUNCTION ---
def get_model_prediction(df_yolo, df_people, df_pico):
    """Get prediction from all models using data with matching timestamps"""
    try:
        # Check if we have data from all sources
        if df_yolo.empty or df_people.empty or df_pico.empty:
            return None, "Missing data from one or more sources", None
        
        # Make sure timestamps are datetime objects
        for df in [df_yolo, df_people, df_pico]:
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            else:
                return None, "Missing timestamp information in one or more data sources", None
        
        # Get the most recent timestamp as a reference point
        latest_timestamps = [
            df_yolo['Timestamp'].max(),
            df_people['Timestamp'].max(),
            df_pico['Timestamp'].max()
        ]
        reference_time = min(latest_timestamps)  # Use the earliest of the latest timestamps
        
        # Find the closest timestamp in each dataframe to the reference time
        yolo_record = df_yolo.iloc[(df_yolo['Timestamp'] - reference_time).abs().argsort()[0]]
        people_record = df_people.iloc[(df_people['Timestamp'] - reference_time).abs().argsort()[0]]
        pico_record = df_pico.iloc[(df_pico['Timestamp'] - reference_time).abs().argsort()[0]]
        
        # Get the values from the matched records
        depth_input = yolo_record.get('People Count', 0)
        rgb_input = people_record.get('Count', 0)
        pico_input = pico_record.get('Count', 0)
        
        # Calculate time differences to check for synchronization
        time_diff_depth = abs((yolo_record['Timestamp'] - reference_time).total_seconds())
        time_diff_rgb = abs((people_record['Timestamp'] - reference_time).total_seconds())
        time_diff_pico = abs((pico_record['Timestamp'] - reference_time).total_seconds())
        
        # Log the inputs and their timestamps for debugging
        st.sidebar.write("### Model Inputs")
        st.sidebar.write(f"- YOLO Depth: {depth_input} (time diff: {time_diff_depth:.2f}s)")
        st.sidebar.write(f"- YOLO RGB: {rgb_input} (time diff: {time_diff_rgb:.2f}s)")
        st.sidebar.write(f"- Pico: {pico_input} (time diff: {time_diff_pico:.2f}s)")
        
        # Warning if timestamps are too far apart (more than 5 seconds)
        max_time_diff = max(time_diff_depth, time_diff_rgb, time_diff_pico)
        if max_time_diff > 5:
            st.sidebar.warning(f"⚠️ Timestamp mismatch: Data points are up to {max_time_diff:.1f} seconds apart")
        
        # Load MLP model
        mlp_model = load_mlp_model()
        if mlp_model is None:
            return None, "MLP model could not be loaded", None
        
        # Load traditional models
        traditional_models = load_traditional_models()
        if traditional_models is None:
            return None, "Traditional models could not be loaded", None
        
        # Prepare input for MLP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlp_model = mlp_model.to(device)
        input_tensor = torch.tensor([float(depth_input), float(rgb_input), float(pico_input)], 
                                  dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get MLP prediction
        with torch.no_grad():
            mlp_prediction = mlp_model(input_tensor).cpu().numpy()
        
        # Prepare input for traditional models
        input_data = np.array([[depth_input, rgb_input, pico_input]])
        scaled_input = traditional_models['scaler'].transform(input_data)
        
        # Get predictions from all traditional models
        rf_prediction = traditional_models['rf'].predict(scaled_input)
        gb_prediction = traditional_models['gb'].predict(scaled_input)
        xgb_prediction = traditional_models['xgb'].predict(scaled_input)
        ensemble_prediction = traditional_models['ensemble'].predict(scaled_input)
        
        # Combine all predictions
        predictions = {
            'MLP': mlp_prediction[0][0],
            'Random Forest': rf_prediction[0],
            'Gradient Boosting': gb_prediction[0],
            'XGBoost': xgb_prediction[0],
            'Ensemble': ensemble_prediction[0]
        }
        
        return predictions, None, reference_time
    except Exception as e:
        import traceback
        st.sidebar.error(f"Error details: {traceback.format_exc()}")
        return None, f"Error making prediction: {str(e)}", None

# --- MAIN UI TABS ---
yolo_tab, people_tab, pico_tab, prediction_tab = st.tabs(["YOLO Depth", "YOLO RGB", "Pico Hypersonic sensors", "Model Prediction"])

with yolo_tab:
    st.header("YOLO Depth Stream")
    df_yolo = load_yolo_depth()
    if df_yolo.empty:
        st.write("No data under `/yolo-depth`.")
    else:
        st.dataframe(df_yolo, use_container_width=True, height=400)
        # Combined People and Fake counts
        st.subheader("People vs Fake Count Over Time")
        melt1 = df_yolo.reset_index(drop=True).melt(
            id_vars=['Timestamp'],
            value_vars=['People Count', 'Fake Count'],
            var_name='Series', value_name='Value'
        )
        
        if not melt1.empty:
            # Get min and max values for better y-axis scale
            y_min = melt1['Value'].min()
            y_max = melt1['Value'].max()
            
            # Calculate a sensible step for ticks based on the range
            step = max(1, int((y_max - y_min) / 4))
            
            chart1 = alt.Chart(melt1).mark_point(size=60).encode(
                x=alt.X('Timestamp:T', 
                       scale=alt.Scale(nice=True),
                       axis=alt.Axis(title='Time', format='%H:%M:%S', labelAngle=-45)),
                y=alt.Y('Value:Q',
                       scale=alt.Scale(domain=[y_min, y_max], nice=False),
                       axis=alt.Axis(title='Count', format='d', values=list(range(int(y_min), int(y_max) + 1, step)))),
                color=alt.Color('Series:N', scale=alt.Scale(
                    domain=['People Count', 'Fake Count'],
                    range=['#00FF00', 'red']
                ))
            ).properties(width='container', height=400)
            st.altair_chart(chart1, use_container_width=True)

        # Separate Fake Count chart
        st.subheader("Fake Count Over Time")
        if not df_yolo.empty and 'Timestamp' in df_yolo.columns and 'Fake Count' in df_yolo.columns:
            # Ensure timestamps are properly formatted
            valid_data = df_yolo.dropna(subset=['Timestamp', 'Fake Count'])
            
            if not valid_data.empty:
                # Get min and max values for better y-axis scale
                y_min = valid_data['Fake Count'].min()
                y_max = valid_data['Fake Count'].max()
                
                # Calculate a sensible step for ticks based on the range
                step = max(1, int((y_max - y_min) / 4))
                
                chart_fake = alt.Chart(valid_data).mark_point(size=60, color='red').encode(
                    x=alt.X('Timestamp:T', 
                           scale=alt.Scale(nice=True),
                           axis=alt.Axis(title='Time', format='%H:%M:%S', labelAngle=-45)),
                    y=alt.Y('Fake Count:Q', 
                           scale=alt.Scale(domain=[y_min, y_max], nice=False),
                           axis=alt.Axis(title='Fake Count', format='d', values=list(range(int(y_min), int(y_max) + 1, step))))
                ).properties(width='container', height=300)
                st.altair_chart(chart_fake, use_container_width=True)
            else:
                st.write("No valid data for Fake Count chart.")
        else:
            st.write("Missing required columns for Fake Count chart.")

with people_tab:
    st.header("YOLO RGB Stream")
    df_pc = load_people_counts()
    if df_pc.empty:
        st.write("No data under `/people_counts`.")
    else:
        st.dataframe(df_pc, use_container_width=True, height=400)
        st.subheader("Count Over Time")
        
        if not df_pc.empty and 'Count' in df_pc.columns:
            # Get min and max values for better y-axis scale
            y_min = df_pc['Count'].min()
            y_max = df_pc['Count'].max()
            
            # Calculate a sensible step for ticks based on the range
            step = max(1, int((y_max - y_min) / 4))
            
            chart_count = alt.Chart(df_pc).mark_point(size=60, color='orange').encode(
                x=alt.X('Timestamp:T', 
                       scale=alt.Scale(nice=True),
                       axis=alt.Axis(title='Time', format='%H:%M:%S', labelAngle=-45)),
                y=alt.Y('Count:Q',
                       scale=alt.Scale(domain=[y_min, y_max], nice=False),
                       axis=alt.Axis(title='Count', format='d', values=list(range(int(y_min), int(y_max) + 1, step))))
            ).properties(width='container', height=300)
            st.altair_chart(chart_count, use_container_width=True)

        st.subheader("Average Confidence Over Time")
        chart_conf = alt.Chart(df_pc).mark_point(size=60, color='green').encode(
            x=alt.X('Timestamp:T', 
                   scale=alt.Scale(nice=True),
                   axis=alt.Axis(title='Time', format='%H:%M:%S', labelAngle=-45)),
            y=alt.Y('Average Confidence:Q',
                   axis=alt.Axis(title='Confidence'))  # Keep as float for confidence values
        ).properties(width='container', height=300)
        st.altair_chart(chart_conf, use_container_width=True)

with pico_tab:
    st.header("Hypersonic Sensor Stream")
    df_pico = load_pico_count()
    if df_pico.empty:
        st.write("No data under `/people_inside`.")
    else:
        st.dataframe(df_pico, use_container_width=True, height=400)
        st.subheader("Count Over Time")
        
        if not df_pico.empty and 'Count' in df_pico.columns:
            # Get min and max values for better y-axis scale
            y_min = df_pico['Count'].min()
            y_max = df_pico['Count'].max()
            
            # Calculate a sensible step for ticks based on the range
            step = max(1, int((y_max - y_min) / 4))
            
            chart_pico = alt.Chart(df_pico).mark_point(size=60, color="green").encode(
                x=alt.X('Timestamp:T', 
                       scale=alt.Scale(nice=True),
                       axis=alt.Axis(title='Time', format='%H:%M:%S', labelAngle=-45)),
                y=alt.Y('Count:Q',
                       scale=alt.Scale(domain=[y_min, y_max], nice=False),
                       axis=alt.Axis(title='Count', format='d', values=list(range(int(y_min), int(y_max) + 1, step))))
            ).properties(width='container', height=300)
            st.altair_chart(chart_pico, use_container_width=True)

with prediction_tab:
    st.header("Model Prediction")
    
    # Reload the data to ensure we have the most recent values
    df_yolo = load_yolo_depth()
    df_pc = load_people_counts()
    df_pico = load_pico_count()
    
    # Display the inputs that will be used for prediction
    st.subheader("Current Inputs for Prediction")
    
    # Get model prediction with timestamp matching
    predictions, error, reference_time = get_model_prediction(df_yolo, df_pc, df_pico)
    
    if error:
        st.error(error)
        
        # Show the input data even if there's an error
        st.write("Available data:")
        if not df_yolo.empty:
            st.write("YOLO Depth data available")
            st.dataframe(df_yolo[['Timestamp', 'People Count']].head(), use_container_width=True)
        else:
            st.write("No YOLO Depth data")
            
        if not df_pc.empty:
            st.write("YOLO RGB data available")
            st.dataframe(df_pc[['Timestamp', 'Count']].head(), use_container_width=True)
        else:
            st.write("No YOLO RGB data")
            
        if not df_pico.empty:
            st.write("Pico data available")
            st.dataframe(df_pico[['Timestamp', 'Count']].head(), use_container_width=True)
        else:
            st.write("No Pico data")
    
    elif predictions is not None:
        # Display the timestamp used for prediction
        st.info(f"Prediction based on data from: {reference_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create a dataframe showing the matched data points used
        if not df_yolo.empty and not df_pc.empty and not df_pico.empty:
            # Find the data points used for prediction
            yolo_record = df_yolo.iloc[(df_yolo['Timestamp'] - reference_time).abs().argsort()[0]]
            people_record = df_pc.iloc[(df_pc['Timestamp'] - reference_time).abs().argsort()[0]]
            pico_record = df_pico.iloc[(df_pico['Timestamp'] - reference_time).abs().argsort()[0]]
            
            # Create a dataframe with the matched data
            matched_data = pd.DataFrame({
                'Sensor': ['YOLO Depth', 'YOLO RGB', 'Pico Hypersonic'],
                'Value': [
                    yolo_record.get('People Count', 'N/A'),
                    people_record.get('Count', 'N/A'),
                    pico_record.get('Count', 'N/A')
                ],
                'Timestamp': [
                    yolo_record.get('Timestamp', 'N/A'),
                    people_record.get('Timestamp', 'N/A'),
                    pico_record.get('Timestamp', 'N/A')
                ],
                'Time Diff (s)': [
                    abs((yolo_record.get('Timestamp') - reference_time).total_seconds()),
                    abs((people_record.get('Timestamp') - reference_time).total_seconds()),
                    abs((pico_record.get('Timestamp') - reference_time).total_seconds())
                ]
            })
            
            st.write("Data points used for prediction:")
            st.dataframe(matched_data, use_container_width=True)
            
            # Calculate time synchronization metrics
            max_time_diff = matched_data['Time Diff (s)'].max()
            if max_time_diff > 5:
                st.warning(f"⚠️ Large timestamp difference detected: Data points are up to {max_time_diff:.1f} seconds apart")
        
        st.subheader("Model Prediction Results")
        
        # Display predictions from all models
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="MLP Prediction", value=f"{predictions['MLP']:.4f}")
        with col2:
            st.metric(label="Random Forest Prediction", value=f"{predictions['Random Forest']:.4f}")
        with col3:
            st.metric(label="Gradient Boosting Prediction", value=f"{predictions['Gradient Boosting']:.4f}")
        
        col4, col5 = st.columns(2)
        with col4:
            st.metric(label="XGBoost Prediction", value=f"{predictions['XGBoost']:.4f}")
        with col5:
            st.metric(label="Ensemble Prediction", value=f"{predictions['Ensemble']:.4f}")
        
        # Create a dataframe to store historical predictions
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = pd.DataFrame(columns=['Timestamp', 'MLP', 'Random Forest', 
                                                                      'Gradient Boosting', 'XGBoost', 'Ensemble',
                                                                      'Reference Time'])
        
        # Add current prediction to history with reference time
        new_row = pd.DataFrame({
            'Timestamp': [datetime.now()],
            'MLP': [predictions['MLP']],
            'Random Forest': [predictions['Random Forest']],
            'Gradient Boosting': [predictions['Gradient Boosting']],
            'XGBoost': [predictions['XGBoost']],
            'Ensemble': [predictions['Ensemble']],
            'Reference Time': [reference_time]
        })
        st.session_state.prediction_history = pd.concat([new_row, st.session_state.prediction_history]).reset_index(drop=True)
        
        # Keep only the last 50 predictions to avoid memory issues
        if len(st.session_state.prediction_history) > 50:
            st.session_state.prediction_history = st.session_state.prediction_history.head(50)
            
        # Display prediction history
        st.subheader("Prediction History")
        st.dataframe(st.session_state.prediction_history, use_container_width=True)
        
        # Plot predictions over time
        if len(st.session_state.prediction_history) > 1:
            st.subheader("Predictions Over Time")
            
            # Create a line chart for predictions
            prediction_chart = alt.Chart(st.session_state.prediction_history).mark_line().encode(
                x=alt.X('Timestamp:T', 
                       scale=alt.Scale(nice=True),
                       axis=alt.Axis(title='Time', format='%H:%M:%S', labelAngle=-45)),
                y=alt.Y('value:Q',
                       axis=alt.Axis(title='Prediction Value')),
                color=alt.Color('variable:N', 
                              scale=alt.Scale(domain=['MLP', 'Random Forest', 'Gradient Boosting', 
                                                    'XGBoost', 'Ensemble'],
                                            range=['blue', 'green', 'orange', 'red', 'purple']))
            ).transform_fold(
                ['MLP', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'Ensemble'],
                as_=['variable', 'value']
            ).properties(width='container', height=400)
            
            st.altair_chart(prediction_chart, use_container_width=True)
    else:
        st.warning("No prediction available. Ensure all data sources have values and the models are correctly loaded.")
        
        # Show available data for debugging
        st.write("Available data:")
        if not df_yolo.empty:
            st.write("YOLO Depth data:")
            st.dataframe(df_yolo[['Timestamp', 'People Count']].head(), use_container_width=True)
        else:
            st.write("No YOLO Depth data")
            
        if not df_pc.empty:
            st.write("YOLO RGB data:")
            st.dataframe(df_pc[['Timestamp', 'Count']].head(), use_container_width=True)
        else:
            st.write("No YOLO RGB data")
            
        if not df_pico.empty:
            st.write("Pico data:")
            st.dataframe(df_pico[['Timestamp', 'Count']].head(), use_container_width=True)
        else:
            st.write("No Pico data")
    
    # Manual refresh button
    if st.button("🔄 Refresh Prediction", key="refresh_prediction"):
        st.experimental_rerun()

# Manual refresh for entire app
if st.button("🔄 Refresh All Data"):
    st.experimental_rerun()