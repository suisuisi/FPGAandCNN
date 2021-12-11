# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


from a00_common_functions import *
from a02_generate_random_non_number import generate_random_non_number
from a01_model_low_weights_digit_detector import keras_model_low_weights_digit_detector
import glob
import os
import random


gpu_use = 0
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


def bbox1(img):
    a = np.where(img < 10)
    try:
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    except:
        bbox = 0, img.shape[0]-1, 0, img.shape[1]-1
    return bbox


def random_rotate(image, max_angle):
    cols = image.shape[1]
    rows = image.shape[0]

    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst


def add_random_noize(image, prob):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.uniform(0, 1) < prob:
                image[i, j] = random.randint(0, 255)
    return image


def read_additional_dataset_images(type, full_size=True):
    files = glob.glob('./dataset/' + type + '/*/*.png')
    image_list = []
    expected_answ = []
    for f in files:
        answ = int(os.path.basename(os.path.dirname(f)))
        expected_answ.append(answ)
        if full_size is True:
            output_image = prepare_image_from_camera(f)
        else:
            output_image = cv2.imread(f, 0)
        image_list.append(output_image)
    image_list = np.array(image_list)
    expected_answ = np.array(expected_answ)
    return image_list, expected_answ


def batch_generator_train(X_train, Y_train, batch_size, type='train'):
    if type == 'train':
        image_list, expected_answ = read_additional_dataset_images('train', full_size=False)

    while True:
        batch_indexes = np.random.choice(X_train.shape[0], batch_size, replace=False)
        batch_images = X_train[batch_indexes].copy()
        batch_classes = Y_train[batch_indexes].copy()

        # replace some images with background one (class 10)
        for i in np.random.choice(list(range(batch_size)), batch_size // 10, replace=False):
            batch_images[i, :, :, 0] = generate_random_non_number()
            batch_classes[i, :] = 0
            batch_classes[i, 10] = 1

        for i in range(batch_images.shape[0]):
            # Random rotate
            img_rotated = random_rotate(batch_images[i, :, :, 0].copy(), 10)

            # Random crop (only for non-background)
            if batch_classes[i, 10] != 1:
                img = np.zeros((32, 32))
                img[...] = 255
                img[2:-2, 2:-2] = img_rotated
                bb = bbox1(img)
                start_0 = random.randint(0, bb[0])
                end_0 = random.randint(bb[1]+1, img.shape[0])
                start_1 = random.randint(0, bb[2])
                end_1 = random.randint(bb[3]+1, img.shape[1])
                subimg = img[start_0:end_0, start_1:end_1].copy()
                interp_type = random.choice([cv2.INTER_LANCZOS4, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
                sub_enlarge = cv2.resize(subimg, (28, 28), interp_type)
            else:
                sub_enlarge = img_rotated

            # Random intensity change
            rand_intensity = random.randint(-80, 80)
            sub_enlarge = sub_enlarge.astype(np.int16) + rand_intensity
            sub_enlarge[sub_enlarge < 0] = 0
            sub_enlarge[sub_enlarge > 255] = 255
            sub_intencity_change = sub_enlarge.astype(np.uint8)

            # Random noize
            sub_noize = add_random_noize(sub_intencity_change, random.uniform(0, 0.1))

            batch_images[i, :, :, 0] = sub_noize.copy()
            if 0:
                show_resized_image(img, 280, 280)
                show_resized_image(subimg, 280, 280)
                show_resized_image(sub_enlarge, 280, 280)
                show_resized_image(sub_intencity_change, 280, 280)
                show_resized_image(batch_images[i, :, :, 0], 280, 280)
                # cv2.imwrite('mnist_aug_debug_{}.png'.format(i), batch_images[i, :, :, 0])
                print(batch_classes[i])

        # Replace single batch image with additional dataset image
        if type == 'train':
            for i in np.random.choice(list(range(batch_size)), 1, replace=False):
                pos = random.randint(0, expected_answ.shape[0]-1)
                replace_image = image_list[pos].copy()
                replace_answ = expected_answ[pos]

                # Random intensity change
                rand_intensity = random.randint(-80, 80)
                replace_image = replace_image.astype(np.int16) + rand_intensity
                replace_image[replace_image < 0] = 0
                replace_image[replace_image > 255] = 255
                replace_image = replace_image.astype(np.uint8)

                # Random noize
                replace_image = add_random_noize(replace_image, random.uniform(0, 0.1))

                batch_images[i, :, :, 0] = replace_image
                batch_classes[i, :] = 0
                batch_classes[i, replace_answ] = 1

        batch_images /= 256.
        yield batch_images, batch_classes


def evaluate_generator_train(X_test, Y_test, batch_size):
    number_of_batches = X_test.shape[0] // batch_size

    i = 0
    while 1:
        if i >= number_of_batches:
            print('Current {}'.format(i))
            batch_images = X_test[-batch_size:]
            batch_classes = Y_test[-batch_size:]
        else:
            batch_images = X_test[i*batch_size:(i+1)*batch_size]
            batch_classes = Y_test[i*batch_size:(i+1)*batch_size]

        batch_images /= 256.
        i += 1
        yield batch_images, batch_classes


if __name__ == '__main__':
    model = keras_model_low_weights_digit_detector()
    continue_training = 0
    final_model_path = os.path.join('weights', 'keras_model_low_weights_digit_detector.h5')
    cache_model_path = os.path.join('weights', 'keras_model_low_weights_digit_detector_cache.h5')

    # Train MNIST if no weights available
    if not os.path.isfile(final_model_path) or continue_training == 1:
        from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
        from keras.optimizers import SGD, Adam

        X_train, Y_train, X_test, Y_test = load_mnist_data(type='channel_last')

        # Invert images
        X_train = 255 - X_train
        X_test = 255 - X_test

        # Append class 10 for background
        Y_train = np.concatenate((Y_train, np.zeros((Y_train.shape[0], 1))), axis=1)
        Y_test = np.concatenate((Y_test, np.zeros((Y_test.shape[0], 1))), axis=1)

        optimizer = 'Adam'
        learning_rate = 0.001
        if optimizer == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optim = Adam(lr=learning_rate)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        if continue_training == 1:
            print('Continue training. Loading weights from: {}'.format(cache_model_path))
            model.load_weights(cache_model_path)

        patience = 30
        batch_size = 50
        print('Weights not found on path: {}. Start training from beginning...'.format(final_model_path))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
            ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]

        history = model.fit_generator(generator=batch_generator_train(X_train, Y_train, batch_size, 'train'),
                                      epochs=2000,
                                      steps_per_epoch=200,
                                      validation_data=batch_generator_train(X_test, Y_test, batch_size, 'valid'),
                                      validation_steps=200,
                                      verbose=2,
                                      max_queue_size=20,
                                      callbacks=callbacks)

        save_history(history, final_model_path, columns=('acc', 'val_acc'))
        score = model.evaluate_generator(generator=evaluate_generator_train(X_test, Y_test, batch_size, 'valid'),
                                         steps=X_test.shape[0] // batch_size,
                                         max_queue_size=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        model.load_weights(cache_model_path)
        model.save(final_model_path)
    else:
        print('Skip training and load weights from {}'.format(final_model_path))
        model.load_weights(final_model_path)

    image_list, expected_answ = read_additional_dataset_images('test')
    print('Test images found: {}'.format(image_list.shape[0]))

    image_list = np.array(image_list, dtype=np.float32) / 256.
    image_list = np.expand_dims(image_list, axis=3)
    preds = model.predict(image_list)
    print('Prediction result')
    pred_answers = np.argmax(preds, axis=1)
    print('Expected answers: {}'.format(expected_answ))
    print('Predicted answers: {}'.format(pred_answers))
    accuracy = 0
    for i in range(len(pred_answers)):
        if pred_answers[i] == expected_answ[i]:
            accuracy += 1
    print('Accuracy: {:.2f}%'.format(100*accuracy / len(pred_answers)))

'''
Accuracy: 97.39%
'''