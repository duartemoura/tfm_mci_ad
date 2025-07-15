"""
Analyze time to conversion and MMSE scores for MCI patients.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_utils import load_pickle, load_csv
from scripts.utils.visualization_utils import (
    plot_metrics_comparison, plot_correlation_matrix
)
from scripts.utils.config import RESULTS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_time_to_conversion(data: pd.DataFrame,
                             save_dir: Path) -> None:
    """
    Analyze time to conversion for MCI patients.
    
    Args:
        data: DataFrame containing time to conversion data
        save_dir: Directory to save analysis results
    """
    try:
        # Create Kaplan-Meier estimator
        kmf = KaplanMeierFitter()
        
        # Fit for converters and non-converters
        kmf.fit(
            data[data['converted'] == 1]['time_to_conversion'],
            data[data['converted'] == 1]['converted'],
            label='Converters'
        )
        
        # Plot survival curve
        plt.figure(figsize=(10, 6))
        kmf.plot()
        plt.title('Time to Conversion - Kaplan-Meier Estimate')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival Probability')
        plt.grid(True)
        plt.savefig(str(save_dir / 'time_to_conversion_km.png'))
        plt.close()
        
        # Calculate median time to conversion
        median_time = kmf.median_survival_time_
        logger.info(f"Median time to conversion: {median_time:.2f} months")
        
        # Save summary statistics
        stats_df = pd.DataFrame({
            'median_time': [median_time],
            'mean_time': [data[data['converted'] == 1]['time_to_conversion'].mean()],
            'std_time': [data[data['converted'] == 1]['time_to_conversion'].std()]
        })
        stats_df.to_csv(save_dir / 'time_to_conversion_stats.csv', index=False)
        
    except Exception as e:
        logger.error(f"Error analyzing time to conversion: {str(e)}")
        raise

def analyze_mmse_scores(data: pd.DataFrame,
                       save_dir: Path) -> None:
    """
    Analyze MMSE scores for MCI patients.
    
    Args:
        data: DataFrame containing MMSE scores
        save_dir: Directory to save analysis results
    """
    try:
        # Create figure for MMSE score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='mmse_score', hue='converted',
                    multiple='stack', bins=30)
        plt.title('MMSE Score Distribution')
        plt.xlabel('MMSE Score')
        plt.ylabel('Count')
        plt.savefig(str(save_dir / 'mmse_distribution.png'))
        plt.close()
        
        # Calculate statistics by conversion status
        stats_by_conversion = data.groupby('converted')['mmse_score'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        # Perform t-test
        converter_scores = data[data['converted'] == 1]['mmse_score']
        non_converter_scores = data[data['converted'] == 0]['mmse_score']
        t_stat, p_value = stats.ttest_ind(converter_scores, non_converter_scores)
        
        # Add t-test results to statistics
        stats_by_conversion['t_statistic'] = t_stat
        stats_by_conversion['p_value'] = p_value
        
        # Save statistics
        stats_by_conversion.to_csv(save_dir / 'mmse_stats.csv', index=False)
        
        # Create box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x='converted', y='mmse_score')
        plt.title('MMSE Scores by Conversion Status')
        plt.xlabel('Converted to AD')
        plt.ylabel('MMSE Score')
        plt.savefig(str(save_dir / 'mmse_boxplot.png'))
        plt.close()
        
    except Exception as e:
        logger.error(f"Error analyzing MMSE scores: {str(e)}")
        raise

def analyze_correlations(data: pd.DataFrame,
                        save_dir: Path) -> None:
    """
    Analyze correlations between different variables.
    
    Args:
        data: DataFrame containing all variables
        save_dir: Directory to save analysis results
    """
    try:
        # Select numeric columns for correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_data = data[numeric_cols]
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(str(save_dir / 'correlation_matrix.png'))
        plt.close()
        
        # Save correlation matrix
        corr_matrix.to_csv(save_dir / 'correlation_matrix.csv')
        
    except Exception as e:
        logger.error(f"Error analyzing correlations: {str(e)}")
        raise

def main():
    """Main function to run the time-to-conversion analysis pipeline."""
    logger.info("Starting time-to-conversion analysis pipeline...")
    
    try:
        # Create results directory
        results_dir = RESULTS_DIR / 'time_to_conversion'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        data = load_csv('mci_conversion_data.csv')
        
        # Analyze time to conversion
        analyze_time_to_conversion(data, results_dir)
        
        # Analyze MMSE scores
        analyze_mmse_scores(data, results_dir)
        
        # Analyze correlations
        analyze_correlations(data, results_dir)
        
    except Exception as e:
        logger.error(f"Error in time-to-conversion analysis pipeline: {str(e)}")
        sys.exit(1)
    
    logger.info("Time-to-conversion analysis pipeline completed")

if __name__ == "__main__":
    main() 