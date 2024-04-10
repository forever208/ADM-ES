import math
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy import stats
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black", linewidth=0.5)

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax, linewidth=0.5)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax, linewidth=0.5)
    axins.add_artist(con)


def pred_eps_l2_norm():
    eps_standard = [np.sqrt(3 * 32 * 32) for i in range(20)]

    """baseline 20steps"""
    eps_l2_norm_training = [35.29843863, 50.28361829, 52.17941961, 53.09097071, 53.59760958, 53.94720221,
                            54.19320703, 54.40821599, 54.56598994, 54.67631154, 54.79086585, 54.86828001,
                            54.96123616, 55.01572878, 55.09332716, 55.15311814, 55.24199719, 55.31557025,
                            55.36786811, 55.41042178]

    eps_l2_norm_sampling = [43.63399923, 52.87061078, 53.72790192, 54.25599827, 54.52366521, 54.70347118,
                            54.83050463, 54.93551159, 55.01418977, 55.06750317, 55.13015985, 55.15283191,
                            55.20506339, 55.2212521, 55.26116489, 55.27719684, 55.32652787, 55.39436218,
                            55.50131555, 55.41109191]

    EDM_l2_norm_training_heun = [23.754921551453798, 21.11153790569898, 41.735277494861656, 42.54055647899334,
                                 49.91392107446245,
                                 50.506342602811856, 53.46900116141852, 53.67029221581245, 54.671693260069034,
                                 54.71438351876444,
                                 55.11698827846194, 55.113190203717565, 55.30707886904951, 55.28788998634221,
                                 55.37917590880489,
                                 55.3811884184169, 55.41231774480642, 55.39886211552898, 55.41891336471381,
                                 55.41766881980681, 55.434242991042964]

    EDM_l2_norm_sampling_heun = [27.647810134653866, 20.8285964302599, 40.860860065343445, 41.93623346759322,
                                 49.40401763186074,
                                 50.1535390659888, 53.32594576266698, 53.59227173710573, 54.71903587066148,
                                 54.72782273824431,
                                 55.15909726537951, 55.12187793718049, 55.324175040437304, 55.29056188157708,
                                 55.38447181410557,
                                 55.37227260208408, 55.40370466031838, 55.39554608461013, 55.415565574697396,
                                 55.40925158360553, 55.42581217914655]

    EDM_l2_norm_training_euler = [23.770714853405995, 35.05111567221983, 41.73714876857001, 46.4043049778302,
                                  49.90699735282854, 52.13842205794809,
                                  53.47240132646554, 54.22511240700536, 54.675509226274066, 54.933641499556025,
                                  55.11296802631953,
                                  55.240813360018066, 55.30498477623677, 55.354860914906986, 55.39180100037825,
                                  55.39655084339965,
                                  55.412290002924784, 55.41968399931553, 55.42873512253551, 55.42774883239436,
                                  55.43372604959194]

    EDM_l2_norm_sampling_euler = [26.310497220786253, 38.59913223388347, 43.99259999335889, 47.832289739980304,
                                  50.693515236934644,
                                  52.52688465441577, 53.66895271502235, 54.308085769986135, 54.704850890553736,
                                  54.93432348855676, 55.09913071055969,
                                  55.22385113485953, 55.28575216405148, 55.33415255668764, 55.36679271954262,
                                  55.38451329612832, 55.39446917775656,
                                  55.403600774644744, 55.41157928238821, 55.41886880397863, 55.42581217914655]

    """linear scaler"""
    eps_l2_norm_sampling_after_solution = [38.26876014, 51.78161165, 53.1982024, 54.15026375, 54.79914334, 55.31869582,
                                           55.74872213, 56.12252704, 56.44371063, 56.70786108, 56.95174074, 57.13749564,
                                           57.31300866, 57.41722054, 57.49577374, 57.49491701, 57.40109353, 57.0187824,
                                           55.55357899, 55.41087023]

    sampling_scaler = [1., 1.00032392, 1.00128684, 1.0028625, 1.00500791,
                       1.00766456, 1.01075998, 1.01420973, 1.01791972, 1.02178874,
                       1.02571126, 1.02958028, 1.03329027, 1.03674002, 1.03983544,
                       1.04249209, 1.0446375, 1.04621316, 1.04717608, 1.0475]

    eps_l2_norm_sampling_after_scaling = [38.26876014, 51.76484398, 53.12983281, 53.99570106, 54.52608163,
                                          54.89792736, 55.15525271, 55.33621437, 55.45006106, 55.49861616,
                                          55.52414501, 55.49591105, 55.46651345, 55.38246757, 55.2931469,
                                          55.15141799, 54.94833713, 54.50015788, 53.05084794, 52.89820547]

    """constant scaler 1.017"""
    eps_l2_norm_sampling_after_solution = [38.48242495, 52.58720926, 53.89897816, 54.64376444, 55.05046381, 55.32282327,
                                           55.51763311, 55.66666056, 55.78215237, 55.86391989, 55.94569219, 55.98696034,
                                           56.05111313, 56.07688379, 56.11738577, 56.12199468, 56.13002538, 56.06422888,
                                           55.54365596, 55.4121359]

    eps_l2_norm_sampling_after_scaling = [37.83915924, 51.70817036, 52.99801196, 53.73034852, 54.13024957,
                                          54.39805631, 54.58960974, 54.73614608, 54.84970735, 54.93010805,
                                          55.01051346, 55.05109178, 55.1141722, 55.13951208, 55.17933704,
                                          55.18386891, 55.19176537, 55.12706871, 54.6151976, 54.48587601]

    """IP training ablation"""
    eps_ablation_l2_norm_training = [34.89728437, 49.63215012, 51.54882622, 52.45652425, 52.96464353, 53.33719253,
                                     53.5919883, 53.79515455, 53.93714825, 54.06548602, 54.17540601, 54.26858295,
                                     54.3452628, 54.4145531, 54.48291676, 54.54925238, 54.61462594, 54.69042235,
                                     54.763527, 54.79300557]

    eps_ablation_l2_norm_sampling = [39.91290718, 51.78662632, 53.02649851, 53.71045786, 54.12840838, 54.43462761,
                                     54.65099241, 54.80632726, 54.91141512, 55.00111602, 55.0634119, 55.12769318,
                                     55.16440777, 55.19784871, 55.2246706, 55.25696318, 55.26525722, 55.21941634,
                                     54.93170103, 54.78859291]

    # Plot the data
    fig, ax = plt.subplots(1, 1)

    ax.plot([i for i in range(1, 22)], EDM_l2_norm_training_heun, linewidth=2, label='training', color='indianred',
            alpha=0.9, marker='^',
            markersize=5,
            linestyle='-', markerfacecolor='mistyrose', markeredgecolor='indianred')
    ax.plot([i for i in range(1, 22)], EDM_l2_norm_sampling_heun, linewidth=2, label='sampling', color='ForestGreen',
            alpha=0.9,
            marker='o', markersize=5,
            linestyle='-', markerfacecolor='honeydew', markeredgecolor='ForestGreen')

    plt.text(5, 33, "Euler step", color="DodgerBlue", size=15)
    plt.arrow(5, 35, -1.5, 5, width=0.05, head_width=0.5, color="DodgerBlue")

    plt.text(5, 25, "Correction step", color="DodgerBlue", size=15)
    plt.arrow(5, 25, -2.3, -4, width=0.05, head_width=0.5, color="DodgerBlue")

    # Customize the grid
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # axis setting
    plt.xlim(1, 21)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylabel("$\|ϵ_{θ} \|_{2}$", fontsize=15)
    plt.xlabel("timestep", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)  # loc='upper right'

    # Show the plot
    plt.savefig('EDM_eps_norm_heun.pdf', bbox_inches='tight')
    plt.show()


def pred_eps_loss_at_each_t():
    eps_loss_ls = [0.6073035597801208, 0.18126867711544037, 0.11619390547275543, 0.08497527241706848,
                   0.06665801256895065,
                   0.05418049916625023, 0.04524359107017517, 0.038286782801151276, 0.03279693424701691,
                   0.028349509462714195,
                   0.024422576650977135, 0.021000565961003304, 0.01777970790863037, 0.014772680588066578,
                   0.011859110556542873,
                   0.008990749716758728, 0.006206754595041275, 0.003538544988259673, 0.0012517013819888234,
                   1.6664536815369502e-05]

    timesteps = [i for i in range(0, 20)]

    scaled_eps_loss_ls = []
    for t in timesteps:
        posterior_mean_coef1 = np.load('./exposure_bias/posterior_mean_coef1_20steps.npz')['arr_0']
        # print(f"posterior_mean_coef1 at {t}: {posterior_mean_coef1[t]}")

        sqrt_alphas_cumprod = np.load('./exposure_bias/sqrt_alphas_cumprod_20steps.npz')['arr_0']
        # print(f"sqrt_alphas_cumprod at {t}: {sqrt_alphas_cumprod[t]}")

        sqrt_one_minus_alphas_cumprod = np.load('./exposure_bias/sqrt_one_minus_alphas_cumprod_20steps.npz')['arr_0']
        # print(f"sqrt_one_minus_alphas_cumprod at {t}: {sqrt_one_minus_alphas_cumprod[t]}")

        scaled_eps_loss = (eps_loss_ls[t]) * posterior_mean_coef1[t] * sqrt_one_minus_alphas_cumprod[t] / \
                          sqrt_alphas_cumprod[t]
        scaled_eps_loss_ls.append(scaled_eps_loss)
    print(eps_loss_ls)

    eps_loss_ls = [0.6073035597801208, 0.18126867711544037, 0.11619390547275543, 0.08497527241706848,
                   0.06665801256895065,
                   0.05418049916625023, 0.04524359107017517, 0.038286782801151276, 0.03279693424701691,
                   0.028349509462714195,
                   0.024422576650977135, 0.021000565961003304, 0.01777970790863037, 0.014772680588066578,
                   0.011859110556542873,
                   0.008990749716758728, 0.006206754595041275, 0.003538544988259673, 0.0012517013819888234,
                   1.6664536815369502e-05]

    scaled_eps_loss_ls = [0.003902176048971225, 0.017342688681541176, 0.014749452096835121, 0.012159322499652995,
                          0.010024758447215789,
                          0.008766966628972672, 0.0076024098603159375, 0.006945535969185877, 0.006354082504276449,
                          0.0058043837533273725,
                          0.005538641333123223, 0.005150121826323143, 0.004971934969262021, 0.0047249949297688675,
                          0.004367203902944327,
                          0.004110484091751661, 0.003587329081416826, 0.002969385021500312, 0.0018439126664257812,
                          0.02789814196167703]

    plt.plot(timesteps, eps_loss_ls, color=(31 / 255, 119 / 255, 180 / 255), marker='o',
             label='eps_loss')
    # plt.plot(timesteps, scaled_eps_loss_ls, color=(255 / 255, 127 / 255, 14 / 255), marker='o',
    #          label='scaled eps_loss')
    plt.legend(prop={'size': 16})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel('timesteps', size=16)
    plt.ylabel('loss', size=16)
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()


def pred_eps_angle_at_each_t():
    timesteps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    angles = [0.8968046, 0.43479434, 0.34278157, 0.2910921, 0.25705105, 0.23082428, 0.21102007, 0.19382381, 0.17913894,
              0.16617152, 0.15433009, 0.14302751, 0.13151301, 0.119666904, 0.10741507, 0.09329783, 0.07744415,
              0.058249533,
              0.03452597, 0.0036580693]

    plt.plot(timesteps, angles, color=(31 / 255, 119 / 255, 180 / 255), marker='o',
             label='angle')
    plt.legend(prop={'size': 16})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel('timesteps', size=16)
    plt.ylabel('angle', size=16)
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()


def plot_eps_norm():
    """baseline 20steps"""
    eps_l2_norm_training = [35.29843863, 50.28361829, 52.17941961, 53.09097071, 53.59760958, 53.94720221,
                            54.19320703, 54.40821599, 54.56598994, 54.67631154, 54.79086585, 54.86828001,
                            54.96123616, 55.01572878, 55.09332716, 55.15311814, 55.24199719, 55.31557025,
                            55.36786811, 55.41042178]

    eps_l2_norm_sampling = [43.63399923, 52.87061078, 53.72790192, 54.25599827, 54.52366521, 54.70347118,
                            54.83050463, 54.93551159, 55.01418977, 55.06750317, 55.13015985, 55.15283191,
                            55.20506339, 55.2212521, 55.26116489, 55.27719684, 55.32652787, 55.39436218,
                            55.50131555, 55.41109191]

    # Plot the data
    fig, ax = plt.subplots(1, 1)

    ax.plot([i for i in range(1, 21)], eps_l2_norm_training, linewidth=2, label='training', color='indianred',
            alpha=0.9, marker='^',
            markersize=5,
            linestyle='-', markerfacecolor='mistyrose', markeredgecolor='indianred')
    ax.plot([i for i in range(1, 21)], eps_l2_norm_sampling, linewidth=2, label='sampling', color='ForestGreen',
            alpha=0.9,
            marker='o', markersize=5,
            linestyle='-', markerfacecolor='honeydew', markeredgecolor='ForestGreen')

    # Customize the grid
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # axis setting
    plt.xlim(1, 21)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylabel("$\|ϵ_{θ} \|_{2}$", fontsize=15)
    plt.xlabel("timestep", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)  # loc='upper right'

    # 绘制缩放图
    axins = ax.inset_axes((0.6, 0.6, 0.4, 0.2))

    # 在缩放图中也绘制主图所有内容，然后
    axins.plot([i for i in range(1, 21)], eps_l2_norm_training, linewidth=2, label='training', color='indianred',
               alpha=0.9, marker='^',
               markersize=5,
               linestyle='-', markerfacecolor='mistyrose', markeredgecolor='indianred')
    axins.plot([i for i in range(1, 21)], eps_l2_norm_sampling, linewidth=2, label='sampling', color='ForestGreen',
               alpha=0.9,
               marker='o', markersize=5,
               linestyle='-', markerfacecolor='honeydew', markeredgecolor='ForestGreen')
    # 局部显示并且进行连线, 根据限制横纵坐标来达成局部显示的目的
    zone_and_linked(ax, axins, 14, 19, [i for i in range(1, 21)], [eps_l2_norm_training, eps_l2_norm_sampling],
                    'bottom')

    plt.text(10, 43, "sampling direction", color="DodgerBlue", size=13)
    plt.arrow(16, 42, -5, 0, width=0.05, head_width=0.5, color="DodgerBlue")

    # Show the plot
    plt.savefig('eps_norm.pdf', bbox_inches='tight')
    plt.show()


def plot_eps_norm_after_solution():
    """cifar10 20steps"""
    eps_l2_norm_training = [35.29843863, 50.28361829, 52.17941961, 53.09097071, 53.59760958, 53.94720221,
                            54.19320703, 54.40821599, 54.56598994, 54.67631154, 54.79086585, 54.86828001,
                            54.96123616, 55.01572878, 55.09332716, 55.15311814, 55.24199719, 55.31557025,
                            55.36786811, 55.41042178]

    eps_l2_norm_sampling = [43.63399923, 52.87061078, 53.72790192, 54.25599827, 54.52366521, 54.70347118,
                            54.83050463, 54.93551159, 55.01418977, 55.06750317, 55.13015985, 55.15283191,
                            55.20506339, 55.2212521, 55.26116489, 55.27719684, 55.32652787, 55.39436218,
                            55.50131555, 55.41109191]

    eps_l2_norm_sampling_after_solution = [38.48242495, 52.58720926, 53.89897816, 54.64376444, 55.05046381, 55.32282327,
                                           55.51763311, 55.66666056, 55.78215237, 55.86391989, 55.94569219, 55.98696034,
                                           56.05111313, 56.07688379, 56.11738577, 56.12199468, 56.13002538, 56.06422888,
                                           55.54365596, 55.4121359]

    eps_l2_norm_sampling_after_solution_divide_1017 = [i / 1.017 for i in eps_l2_norm_sampling_after_solution]
    eps_l2_norm_sampling_after_solution_divide_1017 = [37.8391592428712, 51.70817036381515, 52.9980119567355,
                                                       53.730348515240905, 54.130249567354966, 54.398056312684375,
                                                       54.58960974434612, 54.736146076696166, 54.84970734513275,
                                                       54.93010805309735, 55.010513461160286,
                                                       55.051091779744354, 55.11417220255654, 55.13951208456244,
                                                       55.17933704031466, 55.18386890855458, 55.191765368731566,
                                                       55.12706871189774, 54.61519760078664, 54.48587600786628]

    """lsun 20steps"""
    eps_l2_norm_training = [83.31726, 102.41570827, 105.62351287, 107.1114151, 107.97418658,
                            108.55286055, 108.99579622, 109.32059778, 109.55844365, 109.78967263,
                            109.96173602, 110.10485848, 110.24400555, 110.36581413, 110.43666757,
                            110.52267158, 110.61030753, 110.68530644, 110.75520924, 110.80488703]

    eps_l2_norm_sampling = [106.09914368, 107.08588609, 108.55299993, 109.21295434, 109.61439377,
                            109.89794494, 110.15794557, 110.32999888, 110.43694436, 110.569029,
                            110.64188044, 110.68967626, 110.74807155, 110.80077919, 110.79898802,
                            110.80403516, 110.84529182, 110.87922709, 110.94683397, 110.81287485]

    eps_l2_norm_sampling_after_solution = [95.7056291097923, 103.34545759643919, 106.08162099901088, 107.39184903066273,
                                           108.20763773491593,
                                           108.7682253214639, 109.2263266369931, 109.56317288822949, 109.80303483679526,
                                           110.05080164193869,
                                           110.2254128486647, 110.35842882294759, 110.47792890207715,
                                           110.57783636993078, 110.61237479723049,
                                           110.61442569732938, 110.60982825914937, 110.48619702274976,
                                           109.8680194164194, 109.6041550346192]

    """ffhq128 20steps"""
    eps_l2_norm_training = [136.99359124, 210.32612113, 215.08679541, 217.09944491, 218.1847775,
                            218.91188363, 219.41755862, 219.82308463, 220.10541761, 220.34023485,
                            220.57696496, 220.76855199, 220.90832454, 221.06540179, 221.18195797,
                            221.27314014, 221.39912776, 221.49205019, 221.55122795, 221.64009598]

    eps_l2_norm_sampling = [167.1250783, 215.46103907, 218.5152704, 219.68325809, 220.26349549,
                            220.65613488, 220.91374986, 221.13274122, 221.26457948, 221.36516843,
                            221.47366711, 221.55059776, 221.57196692, 221.61339495, 221.6269189,
                            221.60782544, 221.63841829, 221.65441414, 221.68116212, 221.63756287]

    eps_l2_norm_sampling_after_solution = [123.85933037, 210.09811179, 215.59871313, 217.47263415, 218.41295137,
                                           219.01532147, 219.41400142, 219.74279949, 219.96714755, 220.15216202,
                                           220.3391164, 220.4785176, 220.54423433, 220.60644432, 220.61399192,
                                           220.55548958, 220.49673579, 220.33816489, 220.00265116, 219.88140026]

    # Plot the data
    fig, ax = plt.subplots(1, 1)

    ax.plot([i for i in range(1, 21)], eps_l2_norm_training, linewidth=1.5, label='training', color='indianred',
            alpha=0.9, marker='^',
            markersize=3,
            linestyle='-', markerfacecolor='mistyrose', markeredgecolor='indianred')
    ax.plot([i for i in range(1, 21)], eps_l2_norm_sampling, linewidth=1.5, label='sampling', color='ForestGreen',
            alpha=0.9,
            marker='o', markersize=3,
            linestyle='-', markerfacecolor='honeydew', markeredgecolor='ForestGreen')
    ax.plot([i for i in range(1, 21)], eps_l2_norm_sampling_after_solution, linewidth=1.5,
            label='sampling after Epsilon Scaling', color='DodgerBlue',
            alpha=0.9,
            marker='*', markersize=3,
            linestyle='-', markerfacecolor='honeydew', markeredgecolor='DodgerBlue')

    # Customize the grid
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # axis setting
    plt.xlim(0, 21)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylabel("$\|ϵ_{θ} \|_{2}$", fontsize=15)
    plt.xlabel("timestep", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15, loc='lower right')  # loc='upper right'

    # Show the plot
    plt.savefig('ffhq128_eps_norm_solution.pdf', bbox_inches='tight')
    plt.show()


def plot_eps_norm_diff():
    """baseline 20steps"""
    eps_l2_norm_training = [35.29843863, 50.28361829, 52.17941961, 53.09097071, 53.59760958, 53.94720221,
                            54.19320703, 54.40821599, 54.56598994, 54.67631154, 54.79086585, 54.86828001,
                            54.96123616, 55.01572878, 55.09332716, 55.15311814, 55.24199719, 55.31557025,
                            55.36786811, 55.41042178]

    eps_l2_norm_sampling = [43.63399923, 52.87061078, 53.72790192, 54.25599827, 54.52366521, 54.70347118,
                            54.83050463, 54.93551159, 55.01418977, 55.06750317, 55.13015985, 55.15283191,
                            55.20506339, 55.2212521, 55.26116489, 55.27719684, 55.32652787, 55.39436218,
                            55.50131555, 55.41109191]

    eps_l2_norm_diff = [8.3355606, 2.58699249, 1.5484823100000042, 1.1650275599999986, 0.9260556300000005,
                        0.7562689700000007,
                        0.6372975999999966, 0.5272955999999951, 0.4481998300000001, 0.39119163000000157,
                        0.3392939999999953,
                        0.28455190000000385, 0.24382723000000084, 0.2055233199999975, 0.16783773000000224,
                        0.12407870000000543,
                        0.0845306800000003, 0.07879193000000129, 0.13344743999999764, 0.0006701300000031551,
                        None, None, None, None, None, None, None, None, None, None,
                        None, None, None, None, None, None, None, None, None, None,
                        None, None, None, None, None, None, None, None, None, None,
                        ]

    eps_l2_norm_divide = [1.2361453062378698, 1.0514480178232217, 1.029676112183188, 1.0219439867913465,
                          1.0172779278265716,
                          1.0140186875133224, 1.011759732168041, 1.0096914701282782, 1.0082139044942249,
                          1.0071546821462858,
                          1.0061925285307385, 1.0051860911249295, 1.004436349089569, 1.003735719303508,
                          1.0030464257406813,
                          1.002249713238063, 1.0015301887024335, 1.0014244078049617, 1.0024101964651209,
                          1.0000120939342902,
                          None, None, None, None, None, None, None, None, None, None,
                          None, None, None, None, None, None, None, None, None, None,
                          None, None, None, None, None, None, None, None, None, None,
                          ]
    log_eps_l2_norm_divide = [0.21199791380679056, 0.050168278726740505, 0.029244298599378746, 0.021706682834663326, 0.01713036176371727,
     0.013921334499488174, 0.011691123870683643, 0.009644809065622838, 0.008180353975999154, 0.007129208838093455,
     0.006173433615926777, 0.005172689668498439, 0.004426537500666307, 0.0037287588336724917, 0.003041794788635036,
     0.0022471864222672385, 0.0015290191566327952, 0.0014233942984811245, 0.002407296600182114, 1.2093861159213396e-05,
                              None, None, None, None, None, None, None, None, None, None,
                              None, None, None, None, None, None, None, None, None, None,
                              None, None, None, None, None, None, None, None, None, None,
                              ]

    """""baseline 50steps"""""
    eps_l2_norm_sampling_50steps = [41.80915833, 49.79312149, 51.07178242, 51.82814076, 52.38126966, 52.79249186,
                                    53.10632331, 53.3819528, 53.57075534, 53.75219793, 53.90479526, 54.03003493,
                                    54.1467587, 54.25010728, 54.33218, 54.4028974, 54.46345217, 54.53324028,
                                    54.58406596, 54.63584565, 54.68525055, 54.73075679, 54.77822529, 54.81595796,
                                    54.84736333, 54.87231221, 54.89401274, 54.93459143, 54.96667853, 54.98753313,
                                    55.0070799, 55.04429658, 55.05506612, 55.09410818, 55.10267413, 55.12044842,
                                    55.16080811, 55.17695834, 55.19128803, 55.2235219, 55.24290026, 55.25928112,
                                    55.28022793, 55.3101037, 55.32546517, 55.35803789, 55.38180512, 55.38194787,
                                    55.43106168, 55.41316176]

    eps_l2_norm_training_50steps = [35.29887577, 46.85967104, 49.45134667, 50.70854465, 51.5522724, 52.11047318,
                                    52.53037072, 52.88774623, 53.13470215, 53.35146452, 53.54514025, 53.70378098,
                                    53.85252211, 53.97221045, 54.07470134, 54.16865654, 54.24636692, 54.3323626,
                                    54.39379428, 54.46025429, 54.52175328, 54.57861003, 54.63390453, 54.67208213,
                                    54.71770988, 54.74962935, 54.78284115, 54.82744143, 54.8598653, 54.89396184,
                                    54.91527864, 54.95942716, 54.9763342, 55.01582424, 55.03432846, 55.05539501,
                                    55.10860159, 55.12902908, 55.1452817, 55.18601464, 55.19943163, 55.22686981,
                                    55.24824514, 55.28808469, 55.31290866, 55.3486475, 55.37204577, 55.37800324,
                                    55.41263268, 55.40930235]

    eps_l2_norm_diff_50steps = [6.51028256, 2.933450449999995, 1.6204357499999986, 1.1195961099999963,
                                0.8289972600000013, 0.6820186799999988,
                                0.57595259, 0.4942065700000029, 0.43605318999999554, 0.40073341000000084,
                                0.35965500999999733, 0.3262539500000017,
                                0.294236589999997, 0.2778968300000031, 0.25747866000000386, 0.23424085999999988,
                                0.21708524999999668,
                                0.20087767999999784, 0.19027167999999506, 0.17559135999999853, 0.16349727000000058,
                                0.15214676000000082,
                                0.14432075999999938, 0.14387582999999893, 0.12965344999999928, 0.12268285999999762,
                                0.11117158999999788,
                                0.10714999999999719, 0.1068132300000002, 0.09357128999999986, 0.09180126000000399,
                                0.08486942000000397,
                                0.07873192000000273, 0.07828393999999861, 0.06834566999999936, 0.06505341000000442,
                                0.05220651999999859,
                                0.04792925999999653, 0.046006330000004425, 0.037507259999998155, 0.043468629999999564,
                                0.032411310000000526,
                                0.031982790000000705, 0.02201901000000106, 0.01255651000000313, 0.009390390000000082,
                                0.009759350000003053,
                                0.003944629999999449, 0.018428999999997586, 0.0038594100000040044,
                                ]

    eps_l2_norm_divide_50steps = [1.1844331417923795, 1.0626007478263337, 1.032768283557848, 1.022079042451872,
                                  1.0160807122830147,
                                  1.0130879387267155, 1.010964182854714, 1.0093444437554737, 1.00820656129339,
                                  1.0075111979325286,
                                  1.0067168562510207, 1.006075064810083, 1.005463747629108, 1.0051488873196597,
                                  1.0047615364231248,
                                  1.0043242877885854, 1.0040018394286228, 1.0036972012698744, 1.0034980402179805,
                                  1.003224211166275,
                                  1.00299875297774, 1.00278766278431, 1.0026415970310296, 1.002631614242492,
                                  1.0023694970108277, 1.0022407980009458,
                                  1.0020293140637888, 1.0019543133366309, 1.0019470195454527, 1.0017045825599677,
                                  1.0016716888682622,
                                  1.0015442195158426, 1.001432105671389, 1.0014229349661017, 1.0012418734254143,
                                  1.0011815991872948,
                                  1.000947338863512, 1.0008694014895572, 1.00083427500199, 1.000679651542962,
                                  1.0007874832895267, 1.0005868757384133,
                                  1.0005788924140298, 1.0003982595910756, 1.0002270086730962, 1.000169658888232,
                                  1.0001762504864014,
                                  1.00007123099009, 1.000332577593027, 1.0000696527448698]

    log_eps_l2_norm_divide_50steps = [math.log(i) for i in eps_l2_norm_divide_50steps]

    """""baseline 100steps"""""
    eps_l2_norm_sampling_100steps = [42.43458906, 47.51569086, 48.87978256, 49.78235765, 50.4754254, 51.0012852,
                                     51.44193459, 51.78537974, 52.07002752, 52.32098983, 52.53116601, 52.72204867,
                                     52.88157279, 53.02109014, 53.15806662, 53.27556114, 53.38919897, 53.50275383,
                                     53.5855565, 53.66882864, 53.73066601, 53.8103952, 53.88875474, 53.94947752,
                                     53.99536562, 54.05434154, 54.10975962, 54.14820781, 54.21039317, 54.24699923,
                                     54.28416859, 54.32713715, 54.36045879, 54.40068776, 54.41973241, 54.4605234,
                                     54.50375307, 54.51484847, 54.54071873, 54.5760217, 54.5983212, 54.61432303,
                                     54.64820779, 54.67656523, 54.68899417, 54.71452449, 54.73882918, 54.75781311,
                                     54.76645404, 54.78756832, 54.81049178, 54.8295753, 54.83935297, 54.85507972,
                                     54.87687954, 54.88763322, 54.91051907, 54.90595516, 54.92741662, 54.93929765,
                                     54.96121203, 54.97115394, 54.98769811, 54.99540515, 55.02683512, 55.02505705,
                                     55.03658096, 55.06502674, 55.06505741, 55.08201015, 55.10011523, 55.08436753,
                                     55.11856982, 55.12384079, 55.15623039, 55.14636189, 55.16216795, 55.17180611,
                                     55.18470822, 55.19329244, 55.20939399, 55.22633718, 55.25749819, 55.24695767,
                                     55.25687559, 55.26323619, 55.293665, 55.30133876, 55.32109809, 55.32868891,
                                     55.33662586, 55.36592819, 55.36621094, 55.38418602, 55.40053879, 55.39557125,
                                     55.40160535, 55.43982985, 55.3915925, 55.40716293]

    eps_l2_norm_training_100steps = [35.29986019, 44.35891728, 46.85936021, 48.34823861, 49.3624902, 50.08548272,
                                     50.71015479, 51.15951065, 51.50987113, 51.8282039, 52.08732209, 52.3212339,
                                     52.51016761, 52.68202865, 52.847965, 52.98776889, 53.11994119, 53.24553158,
                                     53.34552198, 53.44030571, 53.51958273, 53.60698034, 53.69997029, 53.76698872,
                                     53.82139733, 53.88516733, 53.953611, 53.99750517, 54.073324, 54.11208186,
                                     54.15340338, 54.19530759, 54.2387265, 54.27790275, 54.30871681, 54.35187106,
                                     54.40038215, 54.41718518, 54.44375263, 54.48587015, 54.51586492, 54.53611679,
                                     54.57396238, 54.60233152, 54.61305283, 54.64270276, 54.67242791, 54.68906448,
                                     54.70616681, 54.72413973, 54.75795846, 54.77545803, 54.78462565, 54.80297906,
                                     54.8338858, 54.83572189, 54.86553945, 54.85489742, 54.87655024, 54.89152003,
                                     54.91649456, 54.92840092, 54.94267266, 54.95479144, 54.98276263, 54.98069578,
                                     54.99784497, 55.02091903, 55.02496774, 55.04389064, 55.0668756, 55.05420363,
                                     55.08602594, 55.08679957, 55.12849576, 55.11968758, 55.13347595, 55.14514511,
                                     55.16452477, 55.17392916, 55.18654589, 55.20953817, 55.23844946, 55.22903426,
                                     55.23922595, 55.24894058, 55.28213275, 55.29629691, 55.30741936, 55.31726758,
                                     55.32948807, 55.35605163, 55.35326289, 55.37186843, 55.38711637, 55.38067073,
                                     55.38626353, 55.41326055, 55.38026496, 55.41405548]

    # Plot the data
    fig, ax = plt.subplots(1, 1)

    ax.plot([i for i in range(5, 51)], log_eps_l2_norm_divide[4:], linewidth=2, label='20 steps', color='indianred',
            alpha=0.9, linestyle='-',
            marker='^', markersize=5,
            markerfacecolor='mistyrose', markeredgecolor='indianred')
    ax.plot([i for i in range(5, 51)], log_eps_l2_norm_divide_50steps[4:], linewidth=2, label='50 steps', color='ForestGreen',
            alpha=0.9, linestyle='-', )
    # marker='o', markersize=5,
    #  markerfacecolor='honeydew', markeredgecolor='ForestGreen')
    # ax.plot([i for i in range(1, 101)], eps_l2_norm_diff_100steps, linewidth=2, label='100steps', color='RoyalBlue',
    #         alpha=0.9, linestyle='-',)
    #         # marker='o', markersize=5,
    #         # markerfacecolor='honeydew', markeredgecolor='DodgerBlue')

    # Customize the grid
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='darkgray')
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='darkgray')

    # remove right and top edges
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # axis setting
    # plt.xlim(1, 51)
    # plt.ylim(1, 1.2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    plt.ylabel(r"$\frac{\|ϵ_{θ}^s \|_{2}}{\|ϵ_{θ}^t \|_{2}}$", fontsize=20, rotation=0, labelpad=22)
    plt.xlabel("timestep", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)  # loc='upper right'
    plt.tight_layout()

    # Show the plot
    plt.savefig('eps_norm_diff.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # pred_eps_l2_norm()
    # pred_eps_loss_at_each_t()
    # pred_eps_angle_at_each_t()
    #
    plot_eps_norm()
    # plot_eps_norm_after_solution()
    # plot_eps_norm_diff()
