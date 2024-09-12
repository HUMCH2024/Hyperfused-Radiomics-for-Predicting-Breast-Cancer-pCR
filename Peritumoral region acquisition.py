import SimpleITK
import numpy as np
from scipy import ndimage

def resize_image_itk(ori_img, target_img, resamplemethod='Linear_interpolation'):
    target_Size = target_img.GetSize()  # Target image size  [x,y,z]
    target_Spacing = target_img.GetSpacing()  # The voxel block size of the target  [x,y,z]
    target_origin = target_img.GetOrigin()  # The starting point of the goal [x,y,z]
    target_direction = target_img.GetDirection()  # The direction of the goal [coronal,sagittal,transverse]=[z,y,x]

    # Resample using the ITK
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # Target images that require resampling
    # Set information for the target image
    resampler.SetSize(target_Size)  # Target image size
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # Set different dypes based on the need to resample images
    if resamplemethod == SimpleITK.sitkNearestNeighbor:
        resampler.SetOutputPixelType(SimpleITK.sitkUInt16)  # Nearest neighbor interpolation for mask, saving uint16
    else:
        resampler.SetOutputPixelType(SimpleITK.sitkFloat32)  # Linear interpolation is used for PET/CT/MRI and other applications, saving float32
    resampler.SetTransform(SimpleITK.Transform(3, SimpleITK.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    try:
        itk_img_resampled = resampler.Execute(ori_img)  # Obtain the resampled image
    except Exception as e:
        print(ori_img)
        print(target_img)
        print('Reshape wrong')

    return itk_img_resampled

def RoiErosionDilationCircle(ori_img,erosionDilationSize):

    img = SimpleITK.ReadImage(ori_img)  # Read nii
    tmp_origin = img.GetOrigin()
    tmp_direction = img.GetDirection()
    tmp_space = img.GetSpacing()
    tmp_n = round(erosionDilationSize/tmp_space[0])
    img_data = SimpleITK.GetArrayFromImage(img)  # Obtain file data

    # Build the nifti image of dilation
    img_data_dilation = img_data
    img_data_dilation = ndimage.binary_dilation(img_data_dilation, structure=np.ones((1, 3, 3)),
                                                iterations=tmp_n).astype(img_data.dtype)              # Dilating

    img_dilation_all = SimpleITK.GetImageFromArray(img_data_dilation)
    img_dilation_all.SetOrigin(tmp_origin)
    img_dilation_all.SetSpacing(tmp_space)
    img_dilation_all.SetDirection(tmp_direction)

    # Build the nifti image of dilation Circle
    img_dilation_circle = SimpleITK.GetImageFromArray(img_data_dilation - img_data)
    img_dilation_circle.SetOrigin(tmp_origin)
    img_dilation_circle.SetSpacing(tmp_space)
    img_dilation_circle.SetDirection(tmp_direction)
    SimpleITK.WriteImage(img_dilation_circle,  ori_img.replace('label.nii', 'label_dilation_circle_'+str(erosionDilationSize)+'.nii'))
    
RoiErosionDilationCircle(r"C:\...\label.nii.gz",   #ROI path
                         5)
