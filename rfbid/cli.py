# rfbid/cli.py
import argparse
import json
import pandas as pd
from rfbid import __version__
from .core import preprocess_olink, pivot_assays, extract_metadata, selection_frequency, validate_markers, compute_classification_metrics, plot_selection_frequency

def run_pipeline():
    parser = argparse.ArgumentParser(description='Random Forest Biomarker Importance Discovery (rfbid)')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))    
    parser.add_argument('--input', required=True, help='Input OLINK NPX CSV file')
    parser.add_argument('--target', required=True, help='Target metadata column to classify')
    parser.add_argument('--panel', required=False, default=None, help='OLINK panel to analyze')
    parser.add_argument('--output', required=False, default='selected_features.txt', help='Output file for selected features')
    parser.add_argument('--metrics_out', required=False, default='metrics.json', help='Output JSON file for metrics')
    parser.add_argument('--plot', action='store_true', help='Show selection frequency plot')
    parser.add_argument('--proba_threshold', type=float, default=0.5, help='Probability threshold for classification')
    parser.add_argument('--positive_label', required=False, default=None, help='(Optional) label in target to treat as positive class (mapped to 1)')
    parser.add_argument('--sep', required=False, default=';', help='CSV separator, default ;')
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep=args.sep)

    data_filtered = preprocess_olink(data, panel=args.panel)
    X_df = pivot_assays(data_filtered)
    # build meta cols excluding common assay columns
    meta_cols = [col for col in data_filtered.columns if col not in ['Index', 'Assay', 'NPX', 'OlinkID', 'UniProt', 'MissingFreq', 'Panel_Version', 'PlateID', 'QC_Warning', 'LOD','Panel']]
    meta_df = extract_metadata(data_filtered, meta_cols)
    
    
    if args.target not in meta_df.columns:
        raise ValueError(f"Target '{args.target}' not found. Available: {meta_df.columns.tolist()}")
    y = meta_df[args.target]

    freq = selection_frequency(X_df, y, n_iter=100, top_k=30, seed=42, rf_kwargs=dict(n_estimators=200, random_state=42, n_jobs=-1))
    selected = freq[freq >= 0.5].index.tolist()

    with open(args.output, 'w') as f:
        for feat in selected:
            f.write(f"{feat}\n")

    val = validate_markers(X_df, y, selected, proba_threshold=args.proba_threshold, positive_label=args.positive_label)

    # val['y_val'] and val['y_pred'] are numeric encoded arrays -> safe to compute metrics

    print(pd.DataFrame.from_dict(val))
    computed = compute_classification_metrics(val['y_val'], val['y_pred'])
    computed['auc'] = val['auc']

    with open(args.metrics_out, 'w') as f:
        json.dump(computed, f, indent=2)

    print('Validation results:')
    for k, v in computed.items():
        print(f"{k}: {v}")

    if args.plot:
        plot_selection_frequency(freq)

def main():
    run_pipeline()

if __name__ == '__main__':
    main()
