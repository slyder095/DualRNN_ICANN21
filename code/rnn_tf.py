import tensorflow as tf
import numpy as np
import time
import psutil
import copy
from IPython.display import display

def data_transform_multiple_inputs_many_to_many(num_batches, seq_len, input_files=[], output_file='', chars_x = [], chars_y = [], desplazar=True, longitud=1000):
    
    for i, f in enumerate(input_files):
        data_dic, dim_i, dim_o, args = data_transform_many_to_many(num_batches, 
                                                      seq_len, 
                                                      input_file=f, 
                                                      output_file=output_file,
                                                      chars_x = chars_x,
                                                      chars_y = chars_y,
                                                      desplazar = desplazar,
                                                      longitud = longitud)

        if i == 0:
            dataset_x_train = copy.deepcopy(data_dic['dataset_x'])
        else:
            dataset_x_train = np.concatenate((dataset_x_train, data_dic['dataset_x']))
        dataset_y_train = data_dic['dataset_y']

    dim_i *= len(input_files)
    
    new_data_dic = {}
    new_data_dic['dataset_x'] = dataset_x_train
    new_data_dic['dataset_y'] = dataset_y_train
    new_data_dic['data_x_as_int'] = data_dic['data_x_as_int']
    new_data_dic['data_y_as_int'] = data_dic['data_y_as_int']
    
    return new_data_dic, dim_i, dim_o, args


def data_transform_many_to_many(num_batches, seq_len, input_file='', output_file='', inputs='', outputs='', chars_x = [], chars_y = [], desplazar=True, longitud=1000):
    
    mf = psutil.virtual_memory().free
    t = time.time()
    
    # Leer los datos y generar utilidad
    if input_file != '': 
        f = open(input_file, 'r')
        data_x_raw = f.read()[:longitud]
        f.close()
    else: data_x_raw = inputs[:longitud]
    if len(chars_x) == 0: chars_x = list(set(data_x_raw))
    char_to_ix = { ch:i for i,ch in enumerate(chars_x) }
    ix_to_char = { i:ch for i,ch in enumerate(chars_x) }

    if output_file != '':
        f = open(output_file, 'r')
        data_y_raw = f.read()[:longitud]
        f.close()
    else: data_y_raw = outputs[:longitud]
    if len(chars_y) == 0: chars_y = list(set(data_y_raw))
    char_to_iy = { ch:i for i,ch in enumerate(chars_y) }
    iy_to_char = { i:ch for i,ch in enumerate(chars_y) }
    
    dataset_len = len(data_x_raw)//num_batches//seq_len * num_batches * seq_len # Para que sea multiplo de num_batches y seq_len
    batch_size = dataset_len//num_batches
    
    # Transformar los datos a int
    data_x_as_int = np.array([char_to_ix[ch] for ch in data_x_raw], dtype=np.uint8)
    data_y_as_int = np.array([char_to_iy[ch] for ch in data_y_raw], dtype=np.uint8)
    
    # Liberar memoria
    data_x_raw = 0
    data_y_raw = 0
    
    # Transformar los datos a one_hot
    data_x_one_hot = np.zeros((len(data_x_as_int), len(chars_x)), dtype=np.uint8)
    data_x_one_hot[np.arange(len(data_x_as_int)), data_x_as_int] = 1
    data_y_one_hot = np.zeros((len(data_y_as_int), len(chars_y)), dtype=np.uint8)
    data_y_one_hot[np.arange(len(data_y_as_int)), data_y_as_int] = 1

    # Redistribuir los datos para poder hacer batch empezando con $ [Esto elimina algunos trozos de cadena]
    data_x_ready_to_batch = []
    data_y_ready_to_batch = []

    index_batches = range(0, dataset_len, batch_size)
    
    dollar_one_hot = np.zeros(len(chars_x))
    dollar_one_hot[char_to_ix['$']] = 1
    
    DATA_X = np.concatenate((data_x_one_hot, data_x_one_hot))
    DATA_Y = np.concatenate((data_y_one_hot, data_y_one_hot))
    
    num_chars_eliminados = 0
    for i in index_batches:
        if desplazar:
            dollar_index = np.where((data_x_one_hot[i:] == dollar_one_hot).all(axis=1))[0][0] + i
        else:
            dollar_index = i
        num_chars_eliminados += dollar_index - i

        data_x_ready_to_batch += DATA_X[dollar_index:dollar_index+batch_size].tolist()
        data_y_ready_to_batch += DATA_Y[dollar_index:dollar_index+batch_size].tolist()
        
    #print " > Se han desplazado " + str(num_chars_eliminados) + " caracteres: " + str(np.round(100. * num_chars_eliminados / len(data_x_raw), decimals=2)) + "%"
    
    data_x_ready_to_batch = np.array(data_x_ready_to_batch, dtype=np.uint8)
    data_y_ready_to_batch = np.array(data_y_ready_to_batch, dtype=np.uint8)
    
    # Liberar memoria
    data_x_one_hot = 0
    DATA_X = 0
    data_y_one_hot = 0
    DATA_Y = 0
    
    # Crear los batches
    dataset_x = []
    dataset_y = []
    
    for i in range(num_batches):
        dataset_x.append(data_x_ready_to_batch[i*batch_size:(i+1)*batch_size])
        dataset_y.append(data_y_ready_to_batch[i*batch_size:(i+1)*batch_size])
    
    data_x_ready_to_batch = 0
    data_y_ready_to_batch = 0
        
    dataset_x = np.transpose(np.array(dataset_x, dtype=np.uint8))
    dataset_y = np.transpose(np.array(dataset_y, dtype=np.uint8))
    
    return {'dataset_x':dataset_x, 'dataset_y':dataset_y, 'data_x_as_int':data_x_as_int, 'data_y_as_int':data_y_as_int}, len(chars_x), len(chars_y), {'ix_to_char':ix_to_char, 'iy_to_char':iy_to_char, 'char_to_ix':char_to_ix, 'char_to_iy':char_to_iy}

def data_transform_many_to_one(num_batches, seq_len, input_file='', output_file='', inputs='', outputs='', chars_x=[], chars_y=[], longitud=1000):
    # Leer los datos y generar utilidad
    if input_file != '': 
        f = open(input_file, 'r')
        data_x_raw = f.read()[:longitud]
        f.close()
    else: data_x_raw = inputs
    if len(chars_x) == 0: chars_x = list(set(data_x_raw))
    char_to_ix = { ch:i for i,ch in enumerate(chars_x) }
    ix_to_char = { i:ch for i,ch in enumerate(chars_x) }

    if output_file != '':
        f = open(output_file, 'r')
        data_y_raw = f.read()[:longitud]
        f.close()
    else: data_y_raw = outputs
    if len(chars_y) == 0: chars_y = list(set(data_y_raw))
    char_to_iy = { ch:i for i,ch in enumerate(chars_y) }
    iy_to_char = { i:ch for i,ch in enumerate(chars_y) } 
    
    
    dataset_len = len(data_x_raw)
    
    # Transformar los datos a int
    data_x_as_int = np.array([char_to_ix[ch] for ch in data_x_raw], dtype=np.uint8)
    data_y_as_int = np.array([char_to_iy[ch] for ch in data_y_raw], dtype=np.uint8)
    
    # Liberar memoria
    data_x_raw = 0
    data_y_raw = 0
    
    # Transformar los datos a one_hot
    data_x_one_hot = np.zeros((len(data_x_as_int), len(chars_x)), dtype=np.uint8)
    data_x_one_hot[np.arange(len(data_x_as_int)), data_x_as_int] = 1
    data_y_one_hot = np.zeros((len(data_y_as_int), len(chars_y)), dtype=np.uint8)
    data_y_one_hot[np.arange(len(data_y_as_int)), data_y_as_int] = 1
    
    
    data_x = []
    data_y = []
    for i in range(dataset_len-seq_len):
        data_x.append(data_x_one_hot[i:i+seq_len])
        data_y.append(data_y_one_hot[i+seq_len])
    

    if len(data_x) % num_batches != 0:
        data_x = np.array(data_x)[:-(len(data_x) % num_batches)]
        data_y = np.array(data_y)[:-(len(data_y) % num_batches)]
    else:
        data_x = np.array(data_x)
        data_y = np.array(data_y)
    
    
    return {'dataset_x':data_x, 'dataset_y':data_y, 'data_x_as_int':data_x_as_int, 'data_y_as_int':data_y_as_int}, len(chars_x), len(chars_y), {'ix_to_char':ix_to_char, 'iy_to_char':iy_to_char, 'char_to_ix':char_to_ix, 'char_to_iy':char_to_iy}
    


        
def train(rnn, epochs, dataset_x, dataset_y, write_tensorboard=False, 
          ruido_progresivo=False, l1reg_mem_progresivo=False, l1reg_cpu_progresivo=False,
          pendiente_ruido=1.0, pendiente_l1reg_mem=1.0, pendiente_l1reg_cpu=1.0,
          ruido_min=0.0, ruido_max=1.0, 
          l1reg_mem_min = 0.0, l1reg_mem_max = 1e-1,
          l1reg_cpu_min = 0.0, l1reg_cpu_max = 1e-1,
          sample_prediction = False, args_prediction = None, step_by_step=True):
    
    if pendiente_ruido != 1.0 and not ruido_progresivo:
        print (" > Warning! El argumento pendiente_ruido no tiene ningun efecto cuando ruido_progresivo = False.")
        
    if pendiente_ruido < 1.0:
        print (" > Warning! Si el argumento pendiente_ruido < 1.0 no alcanzara ruido_max durante el entrenamiento.")
        
    if pendiente_l1reg_mem != 1.0 and not l1reg_mem_progresivo:
        print (" > Warning! El argumento pendiente_l1reg_mem no tiene ningun efecto cuando l1reg_mem_progresivo = False.")
        
    if pendiente_l1reg_mem < 1.0:
        print (" > Warning! Si el argumento pendiente_l1reg_mem < 1.0 no alcanzara l1reg_mem_max durante el entrenamiento.")
        
    if pendiente_l1reg_cpu != 1.0 and not l1reg_cpu_progresivo:
        print (" > Warning! El argumento pendiente_l1reg_cpu no tiene ningun efecto cuando l1reg_cpu_progresivo = False.")
        
    if pendiente_l1reg_cpu < 1.0:
        print (" > Warning! Si el argumento pendiente_l1reg_cpu < 1.0 no alcanzara l1reg_cpu_max durante el entrenamiento.")
    
    smooth_loss = 1.

    shock = False
    limit_percentaje = 0.2

    e = 0
    
    if ruido_progresivo:
        variacion_ruido_epoca = (ruido_max - ruido_min)/epochs
        ruido = ruido_min
        variacion_ruido_epoca *= pendiente_ruido
    else:
        variacion_ruido_epoca = 0
        ruido = rnn.noise_level
    
    if l1reg_mem_progresivo:
        variacion_l1reg_mem_epoca = (l1reg_mem_max - l1reg_mem_min)/epochs
        l1reg_mem = l1reg_mem_min
        variacion_l1reg_mem_epoca *= pendiente_l1reg_mem
    else:
        variacion_l1reg_mem_epoca = 0
        l1reg_mem = rnn.l1reg_value
    
    if l1reg_cpu_progresivo:
        variacion_l1reg_cpu_epoca = (l1reg_cpu_max - l1reg_cpu_min)/epochs
        l1reg_cpu = l1reg_cpu_min
        variacion_l1reg_cpu_epoca *= pendiente_l1reg_cpu
    else:
        variacion_l1reg_cpu_epoca = 0
        l1reg_cpu = rnn.l1reg_value

    #rnn.v_h_prev = np.zeros((rnn.dim_h, rnn.num_batches))
    #rnn.v_c_prev = np.zeros((rnn.dim_h, rnn.num_batches))
    loss_sin_promedio_anterior = 0.
    loss_grafica = []
    while e < epochs:
        loss_sin_promedio = []
        cost_promedio = []
        regu_promedio = []
        rnn.v_h_prev = np.zeros((rnn.dim_h, rnn.num_batches))
        rnn.v_c_prev = np.zeros((rnn.dim_h, rnn.num_batches))

        d1 = display('0.0%', display_id=True)
            
        if step_by_step:
            for i in range(0, len(dataset_x[0]), rnn.seq_len):
                if i % 100 == 0: d1.update(i*100.0/len(dataset_x[0]))
                feed_dict = {}
                feed_dict[rnn.x] = dataset_x[:, i:i+rnn.seq_len]
                feed_dict[rnn.t] = dataset_y[:, i:i+rnn.seq_len]
                feed_dict[rnn.h_prev] = rnn.v_h_prev
                feed_dict[rnn.noise] = [ruido]
                try:
                    feed_dict[rnn.l1reg_mem] = [l1reg_mem]
                    feed_dict[rnn.l1reg_cpu] = [l1reg_cpu]
                except:
                    feed_dict[rnn.l1reg] = [l1reg_mem]
                
                if write_tensorboard:
                    _, loss_sin, costs, regTotal, rnn.v_h_prev, summary = rnn.sess.run((rnn.apply, rnn.loss_sin, 
                                                                                        rnn.loss, rnn.regTotal, 
                                                                                        rnn.h[-1], rnn.merged), 
                                                                                       feed_dict=feed_dict)
                else:
                    _, loss_sin, costs, regTotal, rnn.v_h_prev = rnn.sess.run((rnn.apply, rnn.loss_sin, rnn.loss, 
                                                                               rnn.regTotal, rnn.h[-1]), 
                                                                              feed_dict=feed_dict)
                    
                smooth_loss = smooth_loss * 0.9 + loss_sin * 0.1
                loss_sin_promedio.append(loss_sin)
                cost_promedio.append(costs)
                regu_promedio.append(regTotal)
        else:
            for i in range(len(dataset_x[0])-rnn.seq_len):
                if i % 10 == 0: d1.update(i*100.0/(len(dataset_x[0])-rnn.seq_len))
                feed_dict = {}
                feed_dict[rnn.x] = dataset_x[:, i:i+rnn.seq_len]
                feed_dict[rnn.t] = dataset_y[:, i:i+rnn.seq_len]
                feed_dict[rnn.h_prev] = rnn.v_h_prev
                feed_dict[rnn.noise] = [ruido]
                feed_dict[rnn.l1reg_mem] = [l1reg_mem]
                feed_dict[rnn.l1reg_cpu] = [l1reg_cpu]
                
                if write_tensorboard:
                    _, loss_sin, costs, regTotal, rnn.v_h_prev, summary = rnn.sess.run((rnn.apply, rnn.loss_sin, 
                                                                                        rnn.loss, rnn.regTotal, 
                                                                                        rnn.h[0], rnn.merged), 
                                                                                       feed_dict=feed_dict)
                else:
                    _, loss_sin, costs, regTotal, rnn.v_h_prev = rnn.sess.run((rnn.apply, rnn.loss_sin, rnn.loss, 
                                                                               rnn.regTotal, rnn.h[0]), 
                                                                              feed_dict=feed_dict)
                    
                smooth_loss = smooth_loss * 0.9 + loss_sin * 0.1
                loss_sin_promedio.append(loss_sin)
                cost_promedio.append(costs)
                regu_promedio.append(regTotal)
                
        d1.update('100.0%')
        
        print ("")
        print (" > Epoch " + str(e) + ": " + str(np.mean(cost_promedio)) + " --- " + str(np.mean(cost_promedio)/rnn.num_batches/rnn.seq_len))
        print ("loss epoch mean: " + str(np.mean(loss_sin_promedio)) + " --- 1 batch: " + str(np.mean(loss_sin_promedio)/rnn.num_batches) + " --- mean loss: " + str(np.mean(loss_sin_promedio)/rnn.num_batches/rnn.seq_len))
        print (" regu mean: " + str(np.mean(regu_promedio)))
        print ("    smooth: " + str(smooth_loss) + " --- 1 batch: " + str(smooth_loss/rnn.num_batches) + " --- mean loss: " + str(np.mean(loss_sin_promedio)/rnn.num_batches/rnn.seq_len))
        if ruido_progresivo:
            print ("     ruido: " + str(ruido))
        if l1reg_mem_progresivo:
            print (" l1reg MEM: " + str(l1reg_mem))
        if l1reg_cpu_progresivo:
            print (" l1reg CPU: " + str(l1reg_cpu))
            
        loss_grafica.append(np.mean(cost_promedio))

        e += 1
        loss_sin_promedio_anterior = np.mean(loss_sin_promedio)
        
        if ruido_progresivo:
            ruido += variacion_ruido_epoca
            ruido = np.min([ruido, ruido_max])
            
        if l1reg_mem_progresivo:
            l1reg_mem += variacion_l1reg_mem_epoca
            l1reg_mem = np.min([l1reg_mem, l1reg_mem_max])
            
        if l1reg_cpu_progresivo:
            l1reg_cpu += variacion_l1reg_cpu_epoca
            l1reg_cpu = np.min([l1reg_cpu, l1reg_cpu_max])
            
        if sample_prediction:
            print (rnn.sample_prediction(args_prediction)[-1])
            
    return loss_grafica

def test(rnn, dataset_x, dataset_y):
    
    dataset_len = len(dataset_x[0])//rnn.seq_len * rnn.seq_len
    print(dataset_len)
    salida = np.array(dataset_y[0])[:dataset_len, 0]#, 0]
    h_prev = rnn.v_h_prev

    batch_targets = []
    for i in range(0, dataset_len, rnn.seq_len):
        feed_dict = {}
        feed_dict[rnn.x] = dataset_x[:, i:i+rnn.seq_len]
        feed_dict[rnn.h_prev] = h_prev
        feed_dict[rnn.noise] = [0.0]

        batch_target, h_prev = rnn.sess.run((rnn.y, rnn.h[-1]), feed_dict=feed_dict)
        batch_target = np.array(np.around(batch_target), dtype=np.uint8)
        batch_targets.append(batch_target[:, 0, :])

    batch_targets = np.array(batch_targets)

    # Reorganizamos los datos
    batch_targets_final = []
    for i in range(dataset_len // rnn.seq_len):
        batch_targets_final += batch_targets[i].tolist()
    batch_targets_final = np.array(batch_targets_final)
    
    # Guardamos la prediccion de la red
    mejor_opcion = batch_targets_final.max(axis=1)
    pred = []
    ytrue = []
    for i in range(dataset_len):
        pred.append(np.where(mejor_opcion[i] == batch_targets_final[i])[0][0])
        ytrue.append(np.where(dataset_y[:, i, 0] == 1)[0][0])
    pred = np.array(pred)
    ytrue = np.array(ytrue)
    
    porcentaje_acierto = np.sum((pred == ytrue).astype(int))*100.0/dataset_len
        
    return ytrue, pred, porcentaje_acierto
