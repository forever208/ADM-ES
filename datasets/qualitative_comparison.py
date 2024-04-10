import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats


def show_images():
    x = np.load('../samp_imgnet64_100steps_eps_const_1006/samples_64x64x64x3.npz')['arr_0']
    plt.figure(figsize=(10, 10))
    for i in range(6*6):  # 8*8 or 6*6 or 4*4
        img = x[i+28, :, :, :]
        plt.subplot(6, 6, i + 1)
        plt.imshow(img)
        plt.axis('off')
    # plt.savefig('./ffhq_samples_4.jpg')
    fig = plt.gcf()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.03)  # 调整子图间距
    plt.show()


def pred_x_t_distribution_accumulated_after_solution():
    """
    note that: t in ADM code refers to (t+1) in practical, because ADM starts from index t=0 for training
    """
    timesteps = [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    gt_std = [0.9965907318569104, 0.9866412419612515, 0.9698368766431061, 0.946921649490834, 0.9171708528006858,
     0.8819016821331642, 0.8399986897690623, 0.7923672314492785, 0.7403802256034723, 0.6823932607986238,
     0.6209731989801235, 0.554176709624096, 0.4835957996638227, 0.4111300775486271, 0.3344818221931531,
     0.25703131264953927, 0.17629007788037115, 0.09566472830416303, 0.0064252801356804]

    pred_std = [1.0011720770853572, 0.994943244906608, 0.9864385998419797, 0.9751894449388298, 0.960853983318278,
     0.944043176015839, 0.9243745034133705, 0.9024027626146562, 0.879234031192027, 0.8543720459759546,
     0.8293350285190778, 0.8037454054380456, 0.7789917263629226, 0.7561375205017006, 0.7350799729271481,
     0.7173511878742526, 0.7030104086540329, 0.6934353845620839, 0.6885284489447562]

    std_diff_accumulated = [0.004581345228446776, 0.008302002945356546, 0.01660172319887354, 0.028267795447995736, 0.043683130517592206,
     0.06214149388267476, 0.08437581364430813, 0.11003553116537779, 0.13885380558855476, 0.17197878517733078,
     0.20836182953895432, 0.24956869581394958, 0.29539592669909986, 0.34500744295307345, 0.400598150733995,
     0.4603198752247133, 0.5267203307736616, 0.5977706562579208, 0.6821031688090757]


    pred_std_after_scaling = [1.0012710959417745, 1.0059159680580099, 0.9983507285748298, 0.985866470534044, 0.9690907319115164,
     0.9489447465942552, 0.9252303200579869, 0.8987482273951173, 0.8705349950469099, 0.8400329603658369,
     0.8091106764428938, 0.7774666478508152, 0.7463691629818641, 0.7174481894859733, 0.6906083080296715,
     0.6679193716651449, 0.6493861004904223, 0.6368219791523492, 0.6301691433570037]

    std_diff_after_scaling = [0.004680364084864164, 0.0192747260967584, 0.02851385193172362, 0.038944821043209954, 0.051919879110830625,
     0.06704306446109098, 0.08523163028892455, 0.10638099594583883, 0.13015476944343762, 0.15763969956721302,
     0.18813747746277032, 0.22328993822671916, 0.26277336331804135, 0.3063181119373462, 0.35612648583651846,
     0.41088805901560566, 0.4730960226100512, 0.5411572508481861, 0.6237438632213232]



    # Plot the data
    plt.plot(timesteps, [pred_std[i]**2 - gt_std[i]**2 for i in range(19)], linewidth=2, label='ADM', color='indianred', alpha=0.9, marker='^', markersize=5,
             linestyle='-', markerfacecolor='mistyrose', markeredgecolor='indianred')
    plt.plot(timesteps, [pred_std_after_scaling[i]**2 - gt_std[i]**2 for i in range(19)], linewidth=2, label='ADM-ES', color='ForestGreen', alpha=0.9, marker='o', markersize=5,
             linestyle='-', markerfacecolor='honeydew', markeredgecolor='ForestGreen')

    # Customize the grid
    plt.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    plt.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # axis range
    plt.xlim((1, 19))

    plt.ylabel("$\delta_{t}$", fontsize=15, rotation=0)
    plt.xlabel("timestep", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)  # loc='upper right'

    # Show the plot
    plt.savefig('cifar10_xt_var_error.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    show_images()
    # pred_x_t_distribution_accumulated_after_solution()
