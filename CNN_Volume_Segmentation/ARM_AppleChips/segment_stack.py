from Utils_ARM.trainning import *
from patchify import patchify, unpatchify
import torchvision.transforms as transforms

patch_size = 256

# TODO import model here from saved state
model_dir = '/Users/leonardobertini/Desktop/AlienCoralEnv/WormTubeDataset/MODEL/model_GREY_E10.pth'
model = create_model(params)
model.load_state_dict(torch.load(model_dir))

# TODO import fullsized image stack
stack_dir = '/Volumes/LeoFCP/Bristol_working_scratch/LB_0001/TIFF'
out_dir = '/Volumes/LeoFCP/Bristol_working_scratch/LB_0001/AI_MASK_TEST_2'
masked_pred_dir = os.path.join(out_dir, 'MASKED_OUTPUTS_2')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(masked_pred_dir, exist_ok=True)

image_stack = []
for file in os.listdir(stack_dir):
    if file.endswith('.tif'):
        image_stack.append(file)

# TODO Check if padding is needed
# first get max dim of images in stack to build bounding box in case that's needed
heights, widths = [], []
for image_name in image_stack:
    image = cv2.imread(os.path.join(stack_dir, image_name))
    heights.append(np.shape(image)[0])
    widths.append(np.shape(image)[1])

max_height = max(heights)
max_width = max(widths)

padding_H = patch_size - max_height % patch_size
padding_W = patch_size - max_width % patch_size

# master loop to pad image, break into 256*256 patches, do predictions, then stitch back together and save in directories
for item in range(0,len(image_stack)):
    img = cv2.imread(os.path.join(stack_dir, image_stack[item]), 0)
    old_image_height, old_image_width = img.shape

    # create new image of desired size for padding
    new_image_width = max_width + padding_W
    new_image_height = max_height + padding_H
    color = 0  # padding color
    padded = np.full((new_image_height, new_image_width), color, dtype=np.uint16)  # padding mask

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image --> this is the padded numpy array we pass to patchify
    padded[y_center:y_center + old_image_height,
    x_center:x_center + old_image_width] = img

    padded = padded.astype('uint8')
    # TODO pad to multiple of 256 if necessary
    patches = patchify(padded, (patch_size, patch_size), step=patch_size)

    # cv2.imwrite(os.path.join(stack_dir,"padded_example.tif"), padded)

    # TODO then predict on patches using loaded model
    predicted_patches = []

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            current_patch = patches[i, j, :, :]
            current_patch = np.dstack([current_patch] * 3)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            # get normalized image
            img_normalized = transform_norm(current_patch).float()
            img_normalized = img_normalized.unsqueeze_(0)
            # input = Variable(image_tensor)
            img_normalized = img_normalized.to(device='mps')

            # single patch prediction
            with torch.no_grad():
                model.eval()
                output = model(img_normalized)
                probabilities = torch.sigmoid(output.squeeze(1))
                predicted_mask = (probabilities >= 0.5).float() * 1
                predicted_mask = predicted_mask.cpu().numpy()
                predicted_mask = np.squeeze(predicted_mask, axis=0)
                predicted_patches.append(predicted_mask)

    # TODO Stitch image back together
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size, patch_size))
    reconstructed_mask = unpatchify(predicted_patches_reshaped, padded.shape)  # binary

    reconstructed_mask_grey = (reconstructed_mask * 255).astype('uint8')
    alpha = 0.3
    image_combined = cv2.addWeighted(padded, 1 - alpha, reconstructed_mask_grey, alpha, 0)

    # TODO save image in dir
    prediction_name = image_stack[item].split('.tif')[0] + '_predicted_mask.tiff'
    overlay_name = image_stack[item].split('.tif')[0] + '_overlay.tiff'

    cv2.imwrite(os.path.join(out_dir, prediction_name), reconstructed_mask_grey)
    cv2.imwrite(os.path.join(masked_pred_dir, overlay_name), image_combined)
