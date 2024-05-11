from flask import Flask, render_template, request, flash
import os
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = 'static/upload/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def calculate_psnr(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def add_noise(image, noise_type, std_dev, filename, apply_median=False):
    noisy_image = None
    if noise_type == 'gaussian':
        noise = np.random.normal(0, std_dev, image.shape)
        noisy_image = image + noise.astype(np.uint8)
    elif noise_type == 'salt_and_pepper':
        noise = np.zeros(image.shape, np.uint8)
        prob = 0.05
        salt = np.where(np.random.rand(*image.shape) < prob)
        pepper = np.where(np.random.rand(*image.shape) < prob)
        noise[salt] = 255
        noise[pepper] = 0
        noisy_image = cv2.add(image, noise)
    elif noise_type == 'speckle':
        noise = np.random.lognormal(mean=0, sigma=std_dev, size=image.shape)
        noisy_image = image + image * noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    elif noise_type == 'poisson':
        noise = np.random.poisson(image)
        noisy_image = image + noise.astype(np.uint8)
    else:
        noisy_image = image

    if apply_median:
        noisy_image = cv2.medianBlur(noisy_image, 3)  # Adjust kernel size as needed

    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'noisy_' + noise_type + '_' + filename), noisy_image)
    return noisy_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return render_template('index.html')
        
        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'error')
            return render_template('index.html')
        
        if file:
            if file.mimetype not in ['image/jpeg', 'image/png']:
                flash('Invalid file type. Please upload JPEG or PNG file.', 'error')
                return render_template('index.html')
            
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            original_image = cv2.imread(filepath)

            noisy_gaussian = add_noise(original_image, 'gaussian', 50, filename)
            noisy_salt_pepper = add_noise(original_image, 'salt_and_pepper', 50, filename)
            noisy_speckle = add_noise(original_image, 'speckle', 50, filename)
            noisy_poisson = add_noise(original_image, 'poisson', 50, filename)

            noisy_gaussian_median = add_noise(original_image, 'gaussian', 50, filename, apply_median=True)
            noisy_salt_pepper_median = add_noise(original_image, 'salt_and_pepper', 50, filename, apply_median=True)
            noisy_speckle_median = add_noise(original_image, 'speckle', 50, filename, apply_median=True)
            noisy_poisson_median = add_noise(original_image, 'poisson', 50, filename, apply_median=True)

            psnr_gaussian = calculate_psnr(original_image, noisy_gaussian)
            psnr_salt_pepper = calculate_psnr(original_image, noisy_salt_pepper)
            psnr_speckle = calculate_psnr(original_image, noisy_speckle)
            psnr_poisson = calculate_psnr(original_image, noisy_poisson)

            psnr_gaussian_median = calculate_psnr(original_image, noisy_gaussian_median)
            psnr_salt_pepper_median = calculate_psnr(original_image, noisy_salt_pepper_median)
            psnr_speckle_median = calculate_psnr(original_image, noisy_speckle_median)
            psnr_poisson_median = calculate_psnr(original_image, noisy_poisson_median)

            return render_template('result.html', filename=filename,
                                   psnr_gaussian=psnr_gaussian, psnr_gaussian_median=psnr_gaussian_median,
                                   psnr_salt_pepper=psnr_salt_pepper, psnr_salt_pepper_median=psnr_salt_pepper_median,
                                   psnr_speckle=psnr_speckle, psnr_speckle_median=psnr_speckle_median,
                                   psnr_poisson=psnr_poisson, psnr_poisson_median=psnr_poisson_median)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
