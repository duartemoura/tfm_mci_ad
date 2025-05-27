"""
Split MCI patients into progressive (pMCI) and stable (sMCI) groups.
This script analyzes the conversion status of MCI patients.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, List, Dict
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_utils import (
    load_adni_merge, save_pickle
)
from scripts.utils.config import (
    NEW_DATA_DIR, PICKLE_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_date(date_str: str) -> datetime:
    """
    Parse date string from ADNI merge data.
    
    Args:
        date_str: Date string in format 'MM/DD/YYYY'
        
    Returns:
        datetime object
    """
    try:
        return datetime.strptime(date_str, '%m/%d/%Y')
    except:
        return None

def get_conversion_status(patient_data: pd.DataFrame) -> str:
    """
    Determine if an MCI patient converted to AD.
    
    Args:
        patient_data: DataFrame containing patient's visits
        
    Returns:
        'pMCI' if converted to AD, 'sMCI' if stable
    """
    # Sort visits by date
    patient_data = patient_data.sort_values('EXAMDATE')
    
    # Check if any visit shows conversion to AD
    for _, visit in patient_data.iterrows():
        if visit['DX'] == 'Dementia':
            return 'pMCI'
    
    return 'sMCI'

def split_mci_patients() -> Dict[str, List[str]]:
    """
    Split MCI patients into pMCI and sMCI groups.
    
    Returns:
        Dictionary mapping group names to lists of patient IDs
    """
    # Load ADNI merge data
    adni_merge = load_adni_merge()
    
    # Filter for MCI patients
    mci_patients = adni_merge[adni_merge['DX'] == 'MCI']['PTID'].unique()
    
    pMCI_patients = []
    sMCI_patients = []
    
    for patient_id in mci_patients:
        patient_data = adni_merge[adni_merge['PTID'] == patient_id]
        
        # Get conversion status
        status = get_conversion_status(patient_data)
        
        if status == 'pMCI':
            pMCI_patients.append(patient_id)
        else:
            sMCI_patients.append(patient_id)
    
    # Save results
    results = {
        'pMCI': pMCI_patients,
        'sMCI': sMCI_patients
    }
    save_pickle(results, 'mci_conversion_split.pkl')
    
    return results

def organize_mci_images(split_results: Dict[str, List[str]]) -> None:
    """
    Organize MCI images into pMCI and sMCI directories.
    
    Args:
        split_results: Dictionary mapping group names to lists of patient IDs
    """
    # Create directories
    for group in ['pMCI', 'sMCI']:
        (NEW_DATA_DIR / group).mkdir(parents=True, exist_ok=True)
    
    # Process each MCI image
    mci_dir = NEW_DATA_DIR / 'MCI'
    if not mci_dir.exists():
        raise ValueError(f"MCI directory {mci_dir} does not exist")
    
    for image_path in mci_dir.glob('*.npy'):
        # Extract patient ID from filename
        patient_id = image_path.stem.split('_')[0]
        
        # Determine group
        if patient_id in split_results['pMCI']:
            target_dir = NEW_DATA_DIR / 'pMCI'
        elif patient_id in split_results['sMCI']:
            target_dir = NEW_DATA_DIR / 'sMCI'
        else:
            logger.warning(f"Patient {patient_id} not found in split results")
            continue
        
        # Move image to appropriate directory
        target_path = target_dir / image_path.name
        try:
            os.rename(image_path, target_path)
            logger.info(f"Moved {image_path} to {target_path}")
        except Exception as e:
            logger.error(f"Error moving {image_path}: {str(e)}")

def main():
    """Main function to run the MCI conversion split pipeline."""
    logger.info("Starting MCI conversion split...")
    
    try:
        # Split MCI patients
        split_results = split_mci_patients()
        logger.info(f"Found {len(split_results['pMCI'])} pMCI patients")
        logger.info(f"Found {len(split_results['sMCI'])} sMCI patients")
        
        # Organize images
        organize_mci_images(split_results)
        logger.info("Successfully organized MCI images")
        
    except Exception as e:
        logger.error(f"Error in MCI conversion split pipeline: {str(e)}")
        sys.exit(1)
    
    logger.info("MCI conversion split completed")

if __name__ == "__main__":
    main() 