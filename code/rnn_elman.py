import tensorflow as tf
import numpy as np
import time
import psutil
import copy

from code.rnn_tf import *

class rnn_elman_tf():
    
    def __init__(self, dim_i, dim_h, dim_o, learning_rate, factor_inicializacion=0.01, 
                 seq_len=10, num_batches=1, noise_level=0., regularizationL1=0., shock=0.,
                 optimizer_name='GradientDescent', clipvalue=5.0):
        
        lista_optimizadores = ['GradientDescent', 'Adam', 'RMSProp', 'Adadelta', 'Adagrad', 
                               'Momentum', 'Ftrl']
        if optimizer_name not in lista_optimizadores:
            raise NameError('optimizer_name not valid. Valid optimizers: ' + str(lista_optimizadores))
        
        print (" > Optimizador: " + optimizer_name)
    
        # Matrices de pesos
        self.Wxh = tf.Variable(np.random.randn(dim_h, dim_i)*factor_inicializacion, 
                               dtype=tf.float64, 
                               name = 'Wxh')
        self.Whh = tf.Variable(np.random.randn(dim_h, dim_h)*factor_inicializacion, 
                               dtype=tf.float64, 
                               name = 'Whh')
        self.Why = tf.Variable(np.random.randn(dim_o, dim_h)*factor_inicializacion, 
                               dtype=tf.float64, 
                               name = 'Why')

        # Bias
        self.bh = tf.Variable(np.random.randn(dim_h, 1)*factor_inicializacion,
                              dtype=tf.float64,
                              name = 'bh')
        self.by = tf.Variable(np.random.randn(dim_o, 1)*factor_inicializacion,
                              dtype=tf.float64,
                              name = 'by')

        # Placeholders
        self.x = tf.placeholder(dtype=tf.float64,
                                shape=[dim_i, seq_len, None],
                                name='input')
        self.t = tf.placeholder(dtype=tf.float64,
                                shape=[dim_o, seq_len, None],
                                name='target')
        self.h_prev = tf.placeholder(dtype=tf.float64,
                                     shape=[dim_h, None],
                                     name='h_prev')
        self.x_one_step = tf.placeholder(dtype=tf.float64,
                                shape=[dim_i, None],
                                name='input')
        
        self.noise = tf.placeholder(dtype=tf.float64,
                                    shape=[1],
                                    name = 'Noise')
        self.l1reg = tf.placeholder(dtype=tf.float64,
                                    shape=[1],
                                    name = 'L1reg')

        # Variables necesarias para guardar las variables del UNROLL de la red.
        self.hz = []
        self.h = []
        self.yz = []
        self.y = []
        for i in range(seq_len):
            # Calculos de la red
            if i==0:
                self.hz.append(tf.matmul(self.Wxh, self.x[:, i, :]) + 
                               self.bh + 
                               tf.matmul(self.Whh, self.h_prev) +
                               self.h_prev * self.noise * tf.random_normal([dim_h, num_batches], dtype=tf.float64))
            else:
                self.hz.append(tf.matmul(self.Wxh, self.x[:, i, :]) + 
                               self.bh + 
                               tf.matmul(self.Whh, self.h[-1]) +
                               self.h[-1] * self.noise * tf.random_normal([dim_h, num_batches], dtype=tf.float64))

            self.h.append(tf.tanh(self.hz[-1], name='h'))
            self.yz.append(tf.matmul(self.Why, self.h[-1]) + self.by)
            self.y.append( tf.nn.softmax(tf.transpose(self.yz[-1])))
            
        self.hz_one_step = tf.matmul(self.Wxh, self.x_one_step) + self.bh + tf.matmul(self.Whh, self.h_prev)
        self.h_one_step = tf.tanh(self.hz_one_step)
        self.yz_one_step = tf.matmul(self.Why, self.h_one_step) + self.by
        self.y_one_step = tf.nn.softmax(tf.transpose(self.yz_one_step))

        

        # Regularizacion L1
        regL1Wxh = tf.reduce_sum(tf.abs(self.Wxh), name='regL1_Wxh') * self.l1reg
        regL1Whh = tf.reduce_sum(tf.abs(self.Whh), name='regL1_Whh') * self.l1reg
        regL1Why = tf.reduce_sum(tf.abs(self.Why), name='regL1_Why') * self.l1reg

        self.regTotal = regL1Wxh + regL1Whh + regL1Why
        
        # Funcion de coste
        f_errors = []
        for i in range(seq_len):
            f_errors.append(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.transpose(self.t[:, i, :]),
                                                                       logits=tf.transpose(self.yz[i])))
        self.loss_sin = tf.reduce_sum(f_errors)
        #self.loss_sin = tf.reduce_mean(f_errors)
        self.loss = self.loss_sin + self.regTotal 

        # Optimizador
        if optimizer_name == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        elif optimizer_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        elif optimizer_name == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
        elif optimizer_name == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate)
        elif optimizer_name == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate)
        elif optimizer_name == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = learning_rate*0.1)
        elif optimizer_name == 'Ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate = learning_rate)
            
            
        self.gradients = optimizer.compute_gradients(self.loss)
        clipped = [(tf.clip_by_value(grad, -clipvalue, clipvalue), var) for grad, var in self.gradients]
        self.apply = optimizer.apply_gradients(clipped)
        
        # Shock
        self.shock = [self.Wxh.assign(self.Wxh + tf.random_normal([dim_h, dim_i], dtype=tf.float64) * shock),
                      self.Whh.assign(self.Whh + tf.random_normal([dim_h, dim_h], dtype=tf.float64) * shock),
                      self.Why.assign(self.Why + tf.random_normal([dim_o, dim_h], dtype=tf.float64) * shock)]

        # Inicializador
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        self.l1reg_value = regularizationL1
        self.shock_value = shock
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.dim_i = dim_i
        self.dim_h = dim_h
        self.dim_o = dim_o
        
        # Objeto Saver para salvar y cargar datos:
        self.saver = tf.train.Saver()
        
        #tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('cross_without_L1reg', self.loss_sin)
        tf.summary.scalar('L1reg', self.regTotal)
        tf.summary.scalar('cross_entropy', self.loss)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", tf.get_default_graph())
        #self.writer.close()
        
        
    def __estados_run_one_step__(self, data_input, hprev):
        
        h_prev = hprev

        feed_dict = {}
        feed_dict[self.x_one_step] = data_input[:, 0]
        feed_dict[self.h_prev] = h_prev
        feed_dict[self.noise] = [0.0]
        h, hz, y = self.sess.run((self.h_one_step, self.hz_one_step, self.y_one_step), feed_dict=feed_dict)
        
        return h, hz, y
        
    
    def __estados_run__(self, longitud, data_input):
        
        h_prev = self.v_h_prev
        hs = []
        hzs = []
        ys = []

        for i in range(0, longitud, self.seq_len):
            feed_dict = {}
            feed_dict[self.x] = data_input[:, i:i+self.seq_len]
            feed_dict[self.h_prev] = h_prev
            feed_dict[self.noise] = [0.0]
            y, h, hz = self.sess.run((self.y, self.h, self.hz), feed_dict=feed_dict)
            h_prev = h[-1]
            hs.append(np.array(h)[:, :, 0])
            hzs.append(np.array(hz)[:, :, 0])
            ys.append(np.array(y)[:, 0, :])
            
        hs = np.array(hs)
        hzs = np.array(hzs)
        ys = np.array(ys)
        
        return hs.reshape(hs.shape[0]*hs.shape[1], hs.shape[2]), ys.reshape(ys.shape[0]*ys.shape[1], ys.shape[2]), hzs.reshape(hs.shape[0]*hs.shape[1], hs.shape[2])
                      

    def estados(self, longitud=1000, input_file='', input_files=[], output_file='', inputs='', outputs='', desplazar=True, 
                chars_x = [], chars_y = []):
        
        if len(input_files) > 0:
            data_dic, _, _, transform_dic = data_transform_multiple_inputs_many_to_many(1,
                                                                           self.seq_len,
                                                                           input_files=input_files,
                                                                           output_file=output_file,
                                                                           desplazar=desplazar,
                                                                           chars_x = chars_x,
                                                                           chars_y = chars_y,
                                                                           longitud = longitud)
            
        else:
            data_dic, _, _, transform_dic = data_transform_many_to_many(1, 
                                                           self.seq_len, 
                                                           input_file=input_file, 
                                                           output_file=output_file,
                                                           inputs=inputs,
                                                           outputs=outputs,
                                                           desplazar=desplazar,
                                                           chars_x = chars_x, 
                                                           chars_y = chars_y,
                                                           longitud=longitud)
        
        data_input = data_dic['dataset_x']
        
        dataset_len = len(data_input[0])
        
        if input_file != '':
            input_final = open(input_file, 'r').read()[:np.min([longitud, dataset_len])]
        else:
            input_final = inputs[:longitud]
            
        if output_file != '':
            output_final = open(output_file, 'r').read()[:np.min([longitud, dataset_len])]
        else:
            output_final = outputs[:longitud]
                                                               
        
        hs, ys, hzs = self.__estados_run__(np.min([longitud, dataset_len]), data_input)
        
        return hs, ys, input_final, output_final, data_dic, transform_dic, hzs
   

    def estados_bis(self, hprev, longitud=1000, input_files=[], input_file='', output_file='', inputs='', outputs='', desplazar=True, 
                    chars_x = [], chars_y = []):

        if len(input_files) > 0:
            data_dic, _, _, transform_dic = data_transform_multiple_inputs(1,
                                                                           self.seq_len,
                                                                           input_files=input_files,
                                                                           output_file=output_file,
                                                                           desplazar=desplazar,
                                                                           chars_x = chars_x,
                                                                           chars_y = chars_y,
                                                                           longitud = longitud)
            
        else:
            data_dic, _, _, transform_dic = data_transform(1, 
                                                           self.seq_len, 
                                                           input_file=input_file, 
                                                           output_file=output_file,
                                                           inputs=inputs,
                                                           outputs=outputs,
                                                           desplazar=desplazar,
                                                           chars_x = chars_x, 
                                                           chars_y = chars_y,
                                                           longitud=longitud)
        
        data_input = data_dic['dataset_x']
        
        dataset_len = len(data_input[0])
        
        if input_file != '':
            input_final = open(input_file, 'r').read()[:np.min([longitud, dataset_len])]
        else:
            input_final = inputs[:longitud]
            
        if output_file != '':
            output_final = open(output_file, 'r').read()[:np.min([longitud, dataset_len])]
        else:
            output_final = outputs[:longitud]

        return self.__estados_run_one_step__(data_input, hprev)
    
    def sample_prediction(self, args_prediction, sample_len=100, reset_internal_state=True, 
                          add_diversity=False, temperature=1.0, next_char_ini='$'):
        if reset_internal_state:
            h = np.zeros((self.dim_h, 1))
        else:
            h = self.v_h_prev[:, 0].reshape(self.dim_h, 1)
        ys_str = next_char_ini
        next_char = next_char_ini
        ys = []
        preds = []
        hs = []
        for i in range(sample_len):
            x = np.zeros(self.dim_i)
            x[args_prediction['char_to_ix'][next_char]] = 1.0
            
            feed_dict = {}
            feed_dict[self.x_one_step] = x.reshape(self.dim_i, 1)
            feed_dict[self.h_prev] = h
            
            h, y = self.sess.run((self.h_one_step, self.y_one_step), feed_dict=feed_dict)
            if add_diversity:
                # Parte de Manuel Montanies
                pred = np.log(y+1e-100) / temperature
                exp_pred = np.exp(pred) + 1e-100
                pred = exp_pred / np.sum(exp_pred)
                # Fin de parte de Manuel Montanies
            else:
                pred = y
            choice = np.random.choice(self.dim_o, 1, p=pred.reshape(self.dim_o))[0]
            next_char = args_prediction['iy_to_char'][choice]
            ys.append(np.array(y))
            preds.append(np.array(pred))
            hs.append(np.array(h))
            ys_str += next_char
        return hs, ys, preds, ys_str
    
    def save_model(self, save_file):
        self.saver.save(self.sess, save_file)
        
    def load_model(self, load_file):
        self.saver.restore(self.sess, load_file)
        
        
