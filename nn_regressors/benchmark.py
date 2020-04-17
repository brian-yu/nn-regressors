import subprocess
import os

import tensorflow.compat.v1 as tf
import pandas as pd

from .tf_graph_util import convert_variables_to_constants
from .utils import get_layer_features, preprocess

def benchmark_model(model, cmd=None):
    bench_path = f"{model.name}_benchmark.txt"
    if not os.path.exists(f"{model.name}.pbtxt") and not os.path.exists(bench_path):
        if not os.path.exists(f"{model.name}.pbtxt"):
            print("Saving model...")
#             tf.keras.backend.clear_session()
            sess = tf.keras.backend.get_session()
    #         output_graph_def = tf.graph_util.convert_variables_to_constants(
            output_graph_def = convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(),
                [node.op.name for node in model.outputs])
            tf.io.write_graph(output_graph_def, './', f'{model.name}.pbtxt')
        else:
            print("Retrieving saved model.")
    
    
        if not os.path.exists(bench_path):
            if not cmd:
                input_shape = f"1,{','.join(str(dim) for dim in model.input.shape[1:])}"
                cmd = f'../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph={model.name}.pbtxt --input_layer="{model.input.name}" --input_layer_shape="{input_shape}" --output_layer="{model.output.name}"'
                print(cmd)
            print("Running benchmark...")
            benchmark = subprocess.run([cmd], stderr=subprocess.PIPE, shell=True)
            print("Done.")

            output = benchmark.stderr.decode('unicode_escape')
            split_output = output[output.find('Run Order'):output.find('Top by Computation Time')].split('\n')

            with open(bench_path, 'w') as f:
                f.write("\n".join(split_output[1:-2]))
        else:
            # print("Retrieving saved benchmark results.")
            pass
    else:
        pass
        # print("Retrieving saved model and benchmark results.")
    
    f = open(bench_path)
    benchmark = pd.read_csv(f, sep="\t").rename(columns=lambda x: x.strip())
    benchmark = benchmark.drop(benchmark.columns[0], axis=1)
    benchmark['name'] = benchmark['[Name]'].apply(lambda x: x.split('/')[0])
    return benchmark

def join_benchmark(features, benchmark):
    speed = benchmark[['name', '[avg ms]']].groupby('name').sum()
    mem = benchmark[['name', '[mem KB]']].groupby('name').max()
    
    return features.join(speed, on='name').join(mem, on='name')

def get_benchmark_data(model):
    benchmark = benchmark_model(model)
    features = get_layer_features(model)
    return preprocess(join_benchmark(features, benchmark))