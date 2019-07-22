import os
import numpy as np
from random import shuffle
from collections import namedtuple
from glob import glob
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from tf2_module import build_generator, build_discriminator_classifier, softmax_criterion
from tf2_utils import get_now_datetime, save_midis


class Classifier(object):

    def __init__(self, args):

        self.dataset_A_dir = args.dataset_A_dir
        self.dataset_B_dir = args.dataset_B_dir
        self.sample_dir = args.sample_dir
        self.batch_size = args.batch_size
        self.time_step = args.time_step
        self.pitch_range = args.pitch_range
        self.input_c_dim = args.input_nc  # number of input image channels
        self.sigma_c = args.sigma_c
        self.sigma_d = args.sigma_d
        self.lr = args.lr

        self.model = args.model
        self.generator = build_generator
        self.discriminator = build_discriminator_classifier

        OPTIONS = namedtuple('OPTIONS', 'batch_size '
                                        'time_step '
                                        'input_nc '
                                        'output_nc '
                                        'pitch_range '
                                        'gf_dim '
                                        'df_dim '
                                        'is_training')
        self.options = OPTIONS._make((args.batch_size,
                                      args.time_step,
                                      args.pitch_range,
                                      args.input_nc,
                                      args.output_nc,
                                      args.ngf,
                                      args.ndf,
                                      args.phase == 'train'))

        self.now_datetime = get_now_datetime()

        self._build_model(args)

        print("Initializing classifier...")

    def _build_model(self, args):

        # build classifier
        self.classifier = self.discriminator(self.options,
                                             name='Classifier')

        # optimizer
        self.classifier_optimizer = Adam(self.lr,
                                         beta_1=args.beta1)

        # checkpoints
        model_name = "classifier.model"
        model_dir = "classifier_{}2{}_{}_{}".format(self.dataset_A_dir,
                                                    self.dataset_B_dir,
                                                    self.now_datetime,
                                                    str(self.sigma_c))
        self.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                           model_dir,
                                           model_name)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.checkpoint = tf.train.Checkpoint(classifier_optimizer=self.classifier_optimizer,
                                              classifier=self.classifier)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             self.checkpoint_dir,
                                                             max_to_keep=5)

    def train(self, args):

        # create training list (origin data with corresponding label)
        # Label for A is (1, 0), for B is (0, 1)
        dataA = glob('./datasets/{}/train/*.*'.format(self.dataset_A_dir))
        dataB = glob('./datasets/{}/train/*.*'.format(self.dataset_B_dir))
        labelA = [(1.0, 0.0) for _ in range(len(dataA))]
        labelB = [(0.0, 1.0) for _ in range(len(dataB))]
        data_origin = dataA + dataB
        label_origin = labelA + labelB
        training_list = [pair for pair in zip(data_origin, label_origin)]
        print('Successfully create training list!')

        # create test list (origin data with corresponding label)
        dataA = glob('./datasets/{}/test/*.*'.format(self.dataset_A_dir))
        dataB = glob('./datasets/{}/test/*.*'.format(self.dataset_B_dir))
        labelA = [(1.0, 0.0) for _ in range(len(dataA))]
        labelB = [(0.0, 1.0) for _ in range(len(dataB))]
        data_origin = dataA + dataB
        label_origin = labelA + labelB
        testing_list = [pair for pair in zip(data_origin, label_origin)]
        print('Successfully create testing list!')

        data_test = [np.load(pair[0]) * 2. - 1. for pair in testing_list]
        data_test = np.array(data_test).astype(np.float32)
        gaussian_noise = np.random.normal(0,
                                          self.sigma_c,
                                          [data_test.shape[0],
                                           data_test.shape[1],
                                           data_test.shape[2],
                                           data_test.shape[3]])
        data_test += gaussian_noise
        label_test = [pair[1] for pair in testing_list]
        label_test = np.array(label_test).astype(np.float32).reshape(len(label_test), 2)

        if args.continue_train:
            if self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint):
                print(" [*] Load checkpoint succeeded!")
            else:
                print(" [!] Load checkpoint failed...")

        counter = 1

        for epoch in range(args.epoch):

            # shuffle the training samples
            shuffle(training_list)

            # get the correct batch number
            batch_idx = len(training_list) // self.batch_size

            # learning rate would decay after certain epochs
            self.lr = self.lr if epoch < args.epoch_step else self.lr * (args.epoch-epoch) / (args.epoch-args.epoch_step)

            for idx in range(batch_idx):

                # data samples in batch
                batch = training_list[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_data = [np.load(pair[0]) * 2. - 1. for pair in batch]
                batch_data = np.array(batch_data).astype(np.float32)

                # data labels in batch
                batch_label = [pair[1] for pair in batch]
                batch_label = np.array(batch_label).astype(np.float32).reshape(len(batch_label), 2)

                with tf.GradientTape(persistent=True) as tape:

                    # Origin samples passed through the classifier
                    origin = self.classifier(batch_data,
                                             training=True)
                    test = self.classifier(data_test,
                                           training=True)

                    # loss
                    loss = softmax_criterion(origin, batch_label)

                    # test accuracy
                    test_softmax = tf.nn.softmax(test)
                    test_prediction = tf.equal(tf.argmax(test_softmax, 1), tf.argmax(label_test, 1))
                    test_accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))

                # calculate gradients
                classifier_gradients = tape.gradient(target=loss,
                                                     sources=self.classifier.trainable_variables)

                # apply gradients to the optimizer
                self.classifier_optimizer.apply_gradients(zip(classifier_gradients,
                                                              self.classifier.trainable_variables))

                if idx % 100 == 0:

                    print('=================================================================')
                    print(("Epoch: [%2d] [%4d/%4d] loss: %6.2f, accuracy: %6.2f" %
                           (epoch, idx, batch_idx, loss, test_accuracy)))

                counter += 1

            print('=================================================================')
            print(("Epoch: [%2d] loss: %6.2f, accuracy: %6.2f" % (epoch, loss, test_accuracy)))

            # save the checkpoint per epoch
            self.checkpoint_manager.save(epoch)

    def test(self, args):

        # load the origin samples in npy format and sorted in ascending order
        sample_files_origin = glob('./test/{}2{}_{}_{}_{}/{}/npy/origin/*.*'.format(self.dataset_A_dir,
                                                                                    self.dataset_B_dir,
                                                                                    self.model,
                                                                                    self.sigma_d,
                                                                                    self.now_datetime,
                                                                                    args.which_direction))
        sample_files_origin.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # load the origin samples in npy format and sorted in ascending order
        sample_files_transfer = glob('./test/{}2{}_{}_{}_{}/{}/npy/transfer/*.*'.format(self.dataset_A_dir,
                                                                                        self.dataset_B_dir,
                                                                                        self.model,
                                                                                        self.sigma_d,
                                                                                        self.now_datetime,
                                                                                        args.which_direction))
        sample_files_transfer.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # load the origin samples in npy format and sorted in ascending order
        sample_files_cycle = glob('./test/{}2{}_{}_{}_{}/{}/npy/cycle/*.*'.format(self.dataset_A_dir,
                                                                                  self.dataset_B_dir,
                                                                                  self.model,
                                                                                  self.sigma_d,
                                                                                  self.now_datetime,
                                                                                  args.which_direction))
        sample_files_cycle.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # put the origin, transfer and cycle of the same phrase in one zip
        sample_files = list(zip(sample_files_origin,
                                sample_files_transfer,
                                sample_files_cycle))

        if self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint):
            print(" [*] Load checkpoint succeeded!")
        else:
            print(" [!] Load checkpoint failed...")

        # create a test path to store the generated sample midi files attached with probability
        test_dir_mid = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/mid_attach_prob'.format(self.dataset_A_dir,
                                                                                              self.dataset_B_dir,
                                                                                              self.model,
                                                                                              self.sigma_d,
                                                                                              self.now_datetime,
                                                                                              args.which_direction))
        if not os.path.exists(test_dir_mid):
            os.makedirs(test_dir_mid)

        count_origin = 0
        count_transfer = 0
        count_cycle = 0
        line_list = []

        for idx in range(len(sample_files)):
            print('Classifying midi: ', sample_files[idx])

            # load sample phrases in npy formats
            origin = np.load(sample_files[idx][0])
            transfer = np.load(sample_files[idx][1])
            cycle = np.load(sample_files[idx][2])

            # get the probability for each sample phrase
            origin_softmax = tf.nn.softmax(self.classifier(origin * 2. - 1.,
                                                           training=False))
            transfer_softmax = tf.nn.softmax(self.classifier(transfer * 2. - 1.,
                                                             training=False))
            cycle_softmax = tf.nn.softmax(self.classifier(cycle * 2. - 1.,
                                                          training=False))

            origin_transfer_diff = np.abs(origin_softmax - transfer_softmax)
            content_diff = np.mean((origin * 1.0 - transfer * 1.0) ** 2)

            # labels: (1, 0) for A, (0, 1) for B
            if args.which_direction == 'AtoB':
                line_list.append((idx + 1,
                                  content_diff,
                                  origin_transfer_diff[0][0],
                                  origin_softmax[0][0],
                                  transfer_softmax[0][0],
                                  cycle_softmax[0][0]))

                # for the accuracy calculation
                count_origin += 1 if np.argmax(origin_softmax[0]) == 0 else 0
                count_transfer += 1 if np.argmax(transfer_softmax[0]) == 0 else 0
                count_cycle += 1 if np.argmax(cycle_softmax[0]) == 0 else 0

                # create paths for origin, transfer and cycle samples attached with probability
                path_origin = os.path.join(test_dir_mid, '{}_origin_{}.mid'.format(idx + 1,
                                                                                   origin_softmax[0][0]))
                path_transfer = os.path.join(test_dir_mid, '{}_transfer_{}.mid'.format(idx + 1,
                                                                                       transfer_softmax[0][0]))
                path_cycle = os.path.join(test_dir_mid, '{}_cycle_{}.mid'.format(idx + 1,
                                                                                 cycle_softmax[0][0]))

            else:
                line_list.append((idx + 1,
                                  content_diff,
                                  origin_transfer_diff[0][1],
                                  origin_softmax[0][1],
                                  transfer_softmax[0][1],
                                  cycle_softmax[0][1]))

                # for the accuracy calculation
                count_origin += 1 if np.argmax(origin_softmax[0]) == 1 else 0
                count_transfer += 1 if np.argmax(transfer_softmax[0]) == 1 else 0
                count_cycle += 1 if np.argmax(cycle_softmax[0]) == 1 else 0

                # create paths for origin, transfer and cycle samples attached with probability
                path_origin = os.path.join(test_dir_mid, '{}_origin_{}.mid'.format(idx + 1,
                                                                                   origin_softmax[0][1]))
                path_transfer = os.path.join(test_dir_mid, '{}_transfer_{}.mid'.format(idx + 1,
                                                                                       transfer_softmax[0][1]))
                path_cycle = os.path.join(test_dir_mid, '{}_cycle_{}.mid'.format(idx + 1,
                                                                                 cycle_softmax[0][1]))

            # generate sample MIDI files
            save_midis(origin, path_origin)
            save_midis(transfer, path_transfer)
            save_midis(cycle, path_cycle)

        # sort the line_list based on origin_transfer_diff and write to a ranking txt file
        line_list.sort(key=lambda x: x[2], reverse=True)
        with open(os.path.join(test_dir_mid, 'Rankings_{}.txt'.format(args.which_direction)), 'w') as f:
            f.write('Id  Content_diff  P_O - P_T  Prob_Origin  Prob_Transfer  Prob_Cycle')
            for i in range(len(line_list)):
                f.writelines("\n%5d %5f %5f %5f %5f %5f" % (line_list[i][0],
                                                            line_list[i][1],
                                                            line_list[i][2],
                                                            line_list[i][3],
                                                            line_list[i][4],
                                                            line_list[i][5]))
        f.close()

        # calculate the accuracy
        accuracy_origin = count_origin * 1.0 / len(sample_files)
        accuracy_transfer = count_transfer * 1.0 / len(sample_files)
        accuracy_cycle = count_cycle * 1.0 / len(sample_files)
        print('Accuracy of this classifier on test datasets is :', accuracy_origin, accuracy_transfer, accuracy_cycle)

    def test_famous(self, args):

        song_origin = np.load('./datasets/famous_songs/C2J/merged_npy/Scenes from Childhood (Schumann).npy')
        song_transfer = np.load('./datasets/famous_songs/C2J/transfer/Scenes from Childhood (Schumann).npy')
        print(song_origin.shape, song_transfer.shape)

        if self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint):
            print(" [*] Load checkpoint succeeded!")
        else:
            print(" [!] Load checkpoint failed...")

        sum_origin_A = 0
        sum_origin_B = 0
        sum_transfer_A = 0
        sum_transfer_B = 0

        for idx in range(song_transfer.shape[0]):

            phrase_origin = song_origin[idx]
            phrase_origin = phrase_origin.reshape(1, phrase_origin.shape[0], phrase_origin.shape[1], 1)
            origin_softmax = tf.nn.softmax(self.classifier(phrase_origin * 2. - 1.,
                                                           training=False))

            phrase_transfer = song_transfer[idx]
            phrase_transfer = phrase_transfer.reshape(1, phrase_transfer.shape[0], phrase_transfer.shape[1], 1)
            transfer_softmax = tf.nn.softmax(self.classifier(phrase_transfer * 2. - 1.,
                                                             training=False))

            sum_origin_A += origin_softmax[0][0]
            sum_origin_B += origin_softmax[0][1]
            sum_transfer_A += transfer_softmax[0][0]
            sum_transfer_B += transfer_softmax[0][1]

        print("origin, source:", sum_origin_A / song_transfer.shape[0],
              "target:", sum_origin_B / song_transfer.shape[0])
        print("transfer, source:", sum_transfer_A / song_transfer.shape[0],
              "target:", sum_transfer_B / song_transfer.shape[0])
