"""
Preprocess ADNI data and organize it into appropriate directories.
This script extracts and organizes data from the ADNI database.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_utils import (
    load_nifti_image, preprocess_image, save_pickle
)
from scripts.utils.config import (
    NEW_DATA_DIR, ADNI_MERGE_PATH, IMAGE_SIZE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_xml_metadata(xml_path: str) -> Dict[str, str]:
    """
    Parse XML metadata file from ADNI.
    
    Args:
        xml_path: Path to the XML file
        
    Returns:
        Dictionary containing metadata
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    metadata = {}
    for child in root:
        metadata[child.tag] = child.text
    
    return metadata

def organize_adni_data(adni_root: str) -> None:
    """
    Organize ADNI data into appropriate directories.
    
    Args:
        adni_root: Root directory of ADNI data
    """
    # Create directories for each class
    for class_name in ['AD', 'CN', 'MCI']:
        (NEW_DATA_DIR / class_name).mkdir(parents=True, exist_ok=True)
    
    # Load ADNI merge data for patient information
    adni_merge = pd.read_csv(ADNI_MERGE_PATH)
    
    # Process each subject directory
    for subject_dir in Path(adni_root).glob('*'):
        if not subject_dir.is_dir():
            continue
            
        logger.info(f"Processing subject directory: {subject_dir}")
        
        # Find XML metadata file
        xml_files = list(subject_dir.glob('*.xml'))
        if not xml_files:
            logger.warning(f"No XML metadata found in {subject_dir}")
            continue
            
        metadata = parse_xml_metadata(str(xml_files[0]))
        
        # Get patient information from ADNI merge
        subject_id = metadata.get('Subject ID')
        if not subject_id:
            logger.warning(f"No subject ID found in metadata for {subject_dir}")
            continue
            
        patient_info = adni_merge[adni_merge['PTID'] == subject_id]
        if patient_info.empty:
            logger.warning(f"No patient info found for subject {subject_id}")
            continue
            
        # Determine class based on diagnosis
        diagnosis = patient_info['DX'].iloc[0]
        if diagnosis == 'Dementia':
            class_name = 'AD'
        elif diagnosis == 'CN':
            class_name = 'CN'
        elif diagnosis == 'MCI':
            class_name = 'MCI'
        else:
            logger.warning(f"Unknown diagnosis {diagnosis} for subject {subject_id}")
            continue
        
        # Process NIfTI files
        nifti_files = list(subject_dir.glob('*.nii'))
        if not nifti_files:
            logger.warning(f"No NIfTI files found in {subject_dir}")
            continue
            
        # Process each NIfTI file
        for nifti_file in nifti_files:
            try:
                # Load and preprocess image
                image = load_nifti_image(str(nifti_file))
                processed_image = preprocess_image(image)
                
                # Save processed image
                output_path = NEW_DATA_DIR / class_name / f"{subject_id}_{nifti_file.stem}.npy"
                np.save(output_path, processed_image)
                
                logger.info(f"Saved processed image to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {nifti_file}: {str(e)}")
                continue

def main():
    """Main function to run the preprocessing pipeline."""
    # Check if ADNI root directory is provided
    if len(sys.argv) != 2:
        logger.error("Please provide the ADNI root directory as a command line argument")
        sys.exit(1)
        
    adni_root = sys.argv[1]
    if not os.path.exists(adni_root):
        logger.error(f"ADNI root directory {adni_root} does not exist")
        sys.exit(1)
    
    logger.info("Starting ADNI data preprocessing...")
    organize_adni_data(adni_root)
    logger.info("ADNI data preprocessing completed")

if __name__ == "__main__":
    main() 