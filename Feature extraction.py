import os
from radiomics import featureextractor
import pandas as pd
import SimpleITK as sitk
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

settings = {'binWidth': 32,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None,
            'normalize':True,
           'normalizeScale': 1024}
extractor = featureextractor.RadiomicsFeatureExtractor()

df = pd.DataFrame()
image_path = r'C:\...\image'
roi_path = r'C:\...\roi'

image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(roi_path)
featureVector = extractor.execute(image, mask)
df_add = pd.DataFrame.from_dict(featureVector.values()).T
df_add.columns = featureVector.keys()
df = pd.concat([df,df_add])

df.to_excel(r'C:\...\result.xlsx', index = False)