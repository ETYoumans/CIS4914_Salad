import os
import pyshark
import pandas as pd
import ipaddress
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV

#New model libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Define the malicious IP range and port
MALICIOUS_IP_NETWORK = ipaddress.ip_network("17.57.144.0/24")
MALICIOUS_PORT = 5223
# look at port 443




# Update the feature sets to include rolling average of packet lengths
# and sequence number
def process_pcapng(file_path, interval=60, burst_interval=1, output_dir="processed_data", window_size=10):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the pcapng file
    cap = pyshark.FileCapture(file_path)

    # Initialize variables
    packets = []
    total_packets_count = []
    last_malicious_time = None
    current_total_count = 0
    last_timestamp = None
    first_malicious_packet_flagged = False
    first_timestamp = None
    previous_timestamp = None  # For inter-arrival time calculation
    packet_lengths = []  # To store the packet lengths for rolling average

    # Process each packet
    for packet in cap:
        if 'IP' not in packet or 'TCP' not in packet:
            continue
        # Extract relevant information
        timestamp = float(packet.sniff_timestamp)
        src_ip = packet.ip.src
        dst_ip = packet.ip.dst
        src_port = int(packet.tcp.srcport)
        dst_port = int(packet.tcp.dstport)
        length = int(packet.length)
        seq_num = int(packet.tcp.seq)  # Extract sequence number
        header_len = int(packet.ip.hdr_len) + int(packet.tcp.hdr_len)

        # Extract TCP flags (use binary representation)
        tcp_flags = packet.tcp.flags
        flags_syn = 1 if 'SYN' in tcp_flags else 0
        flags_ack = 1 if 'ACK' in tcp_flags else 0
        flags_fin = 1 if 'FIN' in tcp_flags else 0
        flags_rst = 1 if 'RST' in tcp_flags else 0
        flags_psh = 1 if 'PSH' in tcp_flags else 0
        flags_urg = 1 if 'URG' in tcp_flags else 0

        # Set first_timestamp to the timestamp of the first packet
        if first_timestamp is None:
            first_timestamp = timestamp

        # Calculate relative timestamp based on the first packet
        relative_timestamp = timestamp - first_timestamp

        # Calculate the packet inter-arrival time
        inter_arrival_time = 0
        if previous_timestamp is not None:
            inter_arrival_time = relative_timestamp - previous_timestamp
        
        previous_timestamp = relative_timestamp  # Update the previous timestamp

        # Update total packet count
        current_total_count += 1

        # Update the packet lengths for the rolling average
        packet_lengths.append(length)
        if len(packet_lengths) > window_size:
            packet_lengths.pop(0)  # Keep only the last `window_size` packet lengths

        # Calculate rolling average of packet length
        rolling_avg_length = sum(packet_lengths) / len(packet_lengths)

        # Check if packet is malicious
        is_malicious = 0
        if (ipaddress.ip_address(src_ip) in MALICIOUS_IP_NETWORK or ipaddress.ip_address(dst_ip) in MALICIOUS_IP_NETWORK) and \
           (dst_port == MALICIOUS_PORT or src_port == MALICIOUS_PORT):
            if not first_malicious_packet_flagged:
                first_malicious_packet_flagged = True  # Mark the first malicious packet
            is_malicious = 1
            last_malicious_time = relative_timestamp  # Update the last malicious packet time
        
        # For packets after the first malicious one, check the interval
        elif first_malicious_packet_flagged and relative_timestamp - last_malicious_time >= interval:
            if (ipaddress.ip_address(src_ip) in MALICIOUS_IP_NETWORK or ipaddress.ip_address(dst_ip) in MALICIOUS_IP_NETWORK) and \
               (dst_port == MALICIOUS_PORT or src_port == MALICIOUS_PORT):
                is_malicious = 1
                last_malicious_time = relative_timestamp  # Update the last malicious packet time

        # Update the list of packets with the TCP flags as additional features
        packets.append((relative_timestamp, src_ip, dst_ip, src_port, dst_port, length, inter_arrival_time, rolling_avg_length, seq_num, header_len,
                        flags_syn, flags_ack, flags_fin, flags_rst, flags_psh, flags_urg, is_malicious))

        # Only check the time difference once last_timestamp has been set
        if last_timestamp is not None and relative_timestamp - last_timestamp >= burst_interval:
            # For the 60-second interval, save the total and malicious packet counts
            if relative_timestamp - last_timestamp >= interval:
                total_packets_count.append(current_total_count)
                current_total_count = 0  # Reset total count for the next burst
                last_timestamp = relative_timestamp

        # Set last_timestamp initially to the current packet's relative timestamp
        if last_timestamp is None:
            last_timestamp = relative_timestamp
    
    # Create a DataFrame from the packets list
    df = pd.DataFrame(packets, columns=['relative_timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'length', 'inter_arrival_time', 'rolling_avg_length', 'seq_num', 'header_len',
                                        'flags_syn', 'flags_ack', 'flags_fin', 'flags_rst', 'flags_psh', 'flags_urg', 'is_malicious'])
    
    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")

    return df  # Return DataFrame containing all packets with labels

# Train and evaluate the model with the new feature
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, average_precision_score

def train_and_evaluate(training_file, testing_file, model_type, output_dir="processed_data", test_size=0.2, n_splits=5):
    # Process the training file
    print("Processing training file...")
    train_df = process_pcapng(training_file, output_dir=output_dir)
    print(f"Processed {len(train_df)} packets for training.")

# Separate features and labels for training    
    X_train_full = train_df[['length', 'header_len', 'inter_arrival_time', 'rolling_avg_length', 'seq_num']]
    y_train_full = train_df['is_malicious']

    # train test split
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=test_size, random_state=42)

    # Set hyperparameter grid for each model type
    if model_type == 'RandomForest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None],
        }
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall', n_jobs=-1)

    elif model_type == 'LogisticRegression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None],
        }
        model = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall', n_jobs=-1)

    elif model_type == 'SVC':
        param_grid = {
            'C': [0.1, 1, 10],
            #'kernel': ['linear', 'rbf'],
            'class_weight': ['balanced', None],
        }
        model = SVC(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=2, scoring='recall', n_jobs=-1)

    elif model_type == 'kNearestNeighbors':
        param_grid = {
            'n_neighbors' : [11,49,99,199],
            'weights' : ['uniform','distance'],
        }
        model = KNeighborsClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall',n_jobs=-1)


    else:
        print("Invalid model type. Using RandomForest by default.")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None],
        }
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall', n_jobs=-1)

    # Perform Grid Search to find the best hyperparameters
    print(f"Performing GridSearchCV for {model_type}...")
    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters for {model_type}: {grid_search.best_params_}")
    
    # Get the best model
    best_model = grid_search.best_estimator_

    # test it with the validation set
    print("Testing the best model with the validation set...")
    y_pred = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print("Validation Classification Report:\n", classification_report(y_val, y_pred))
    
    # Process the testing file (i.e., another pcap file with different characteristics to test the model's performance)
    print("Processing testing file...")
    test_df = process_pcapng(testing_file, output_dir=output_dir, interval=120)
    print(f"Processed {len(test_df)} packets for testing.")

    # Separate features and labels for testing
    X_test = test_df[['length', 'header_len', 'inter_arrival_time', 'rolling_avg_length', 'seq_num']]
    y_test = test_df['is_malicious']

    # Evaluate the model on the testing set
    print("Testing Accuracy:", accuracy_score(y_test, best_model.predict(X_test)))
    print("Testing Classification Report:\n", classification_report(y_test, best_model.predict(X_test)))

    # Confusion Matrix for testing set
    cm_test = confusion_matrix(y_test, best_model.predict(X_test))
    print("Testing Confusion Matrix (True Positives, False Positives, True Negatives, False Negatives):")
    print(cm_test)

    # Plot confusion matrix using seaborn heatmap for testing set
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Malicious', 'Malicious'], yticklabels=['Not Malicious', 'Malicious'])
    plt.title(f'{model_type} Testing Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Precision-Recall Curve for testing set
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else best_model.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.2f})')
    plt.title('Precision-Recall Curve (Testing)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process training and testing pcapng files and train a machine learning model.")
    parser.add_argument("training_file", type=str, help="Path to the training pcapng file")
    parser.add_argument("testing_file", type=str, help="Path to the testing pcapng file")
    parser.add_argument("model", type=str, choices=['RandomForest', 'LogisticRegression', 'SVC', 'kNearestNeighbors'], help="The machine learning model to use")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Directory to save the processed .csv files")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the training data to use for validation (default 0.2)")
    args = parser.parse_args()
    print("Arguments parsed:", args)

    # Train and evaluate the model
    print("Training and evaluating the model...")
    train_and_evaluate(args.training_file, args.testing_file, args.model, output_dir=args.output_dir, test_size=args.test_size)