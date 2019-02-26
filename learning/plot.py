import tensorflow as tf
import matplotlib.pyplot as plt


def plot(file_path_list):
    fitness_list = [[] for _ in range(len(file_path_list))]
    for i, event in enumerate(events_file_path_list):
        for e in tf.train.summary_iterator(event):
            for v in e.summary.value:
                if v.tag == 'mean_team_fitness':
                    fitness_list[i].append(v.simple_value)

    for i in range(len(fitness_list)):
        plt.plot(fitness_list[i])
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("R5-P5-Cp3_res90_density_obs125")
    plt.show()


if __name__=="__main__":
    events_file_path_list = \
        ["/home/yt/Research/ccea/learning/Experiments/R5-P5-Cp3__0225-130226/events.out.tfevents.1551128546.aadi-z",
            "/home/yt/Research/ccea/learning/Experiments/R5-P5-Cp3__0225-130232/events.out.tfevents.1551128552.aadi-z"]
    plot(events_file_path_list)
