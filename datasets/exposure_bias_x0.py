import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats


def pred_x_0_error_accumulated():
    """
    note that: t in ADM code refers to (t+1) in practical, because ADM starts from index t=0 for training
    """

    timesteps = []
    x_0_error_std = []

    filename = f"./exposure_bias/x_0_error_accumulate_500steps_IP/x_0_error.npz"
    x = np.load(filename)
    print(f"array keys:{x.files} and each array has shape {x[x.files[0]].shape}")

    for t in x.files:
        t = int(t)
        means = []
        stds = []
        data = x[str(t)]

        # compute the mean and std for each timestep t among 50k predictions
        for c in range(data.shape[1]):
            for h in range(data.shape[2]):
                for w in range(data.shape[3]):
                    pixel_data = data[:, c, h, w]
                    mean = np.mean(pixel_data)
                    std = np.std(pixel_data)
                    means.append(mean)
                    stds.append(std)
        pred_x_0_std = sum(stds)/len(stds)
        print(f"x_0_error_std at {t} steps: {pred_x_0_std}, mean:{sum(means)/len(means)}")

        if int(t) not in timesteps:
            timesteps.append(int(t))
            x_0_error_std.append(pred_x_0_std)
            print(f"timestep {t} added into plot")
            print(f"")

    print(f"timesteps: {timesteps}")
    print(f"x_0_error_std: {x_0_error_std}")

    plt.plot(timesteps, x_0_error_std, color=(31 / 255, 119 / 255, 180 / 255), marker='o', label='x_0_error_std')
    plt.legend(prop={'size': 16})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel('timesteps', size=16)
    plt.ylabel('std', size=16)
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()

    timesteps = [450, 400, 350, 300, 250, 200, 150, 100, 50, 0]

    x_0_error_std = [0.17539888678584248, 0.18473645337022995, 0.19248000752607672, 0.1991320699841405, 0.20503026439594882,
                     0.21030101019520467, 0.21513090903075258, 0.21954661657703886, 0.22346041608640613, 0.22632195357679544]


def pred_x_0_error_at_each_t():
    timesteps = []
    x_0_error_std_ls = []
    scaled_x_0_error_std_ls = []
    for file in ['10']:
        filename = f"./exposure_bias/x_0_error_each_{file}steps/x_0_error_each_{file}steps.npz"
        x = np.load(filename)
        print(f"array keys:{x.files} and each array has shape {x[x.files[0]].shape}")

        for t in x.files:
            t = int(t)
            posterior_mean_coef1 = np.load('./exposure_bias/posterior_mean_coef1.npz')['arr_0']
            print(f"posterior_mean_coef1 at {t}: {posterior_mean_coef1[t]}")

            means = []
            stds = []
            data = x[str(t)]

            # compute the mean and std for each timestep t among 50k predictions
            for c in range(data.shape[1]):
                for h in range(data.shape[2]):
                    for w in range(data.shape[3]):
                        pixel_data = data[:, c, h, w]
                        mean = np.mean(pixel_data)
                        std = np.std(pixel_data)
                        means.append(mean)
                        stds.append(std)
            x_0_error_std = sum(stds) / len(stds)
            print(f"x_0 error at {t} std:{x_0_error_std}, mean:{sum(means) / len(means)}")
            print(f"scaled x_0 error at {t} std:{x_0_error_std * posterior_mean_coef1[t]}")

            if int(t) not in timesteps:
                timesteps.append(int(t))
                x_0_error_std_ls.append(x_0_error_std)
                scaled_x_0_error_std_ls.append(x_0_error_std * posterior_mean_coef1[t])
                print(f"timestep {t} added into plot")
                print(f"")

    print(f"timesteps: {timesteps}")
    print(f"x_0_error_std: {x_0_error_std_ls}")
    print(f"scaled x_0_error_std: {scaled_x_0_error_std_ls}")

    # plt.plot(timesteps, x_0_error_std_ls, color=(31 / 255, 119 / 255, 180 / 255), marker='o', label='x_0_error_std')
    plt.plot(timesteps, scaled_x_0_error_std_ls, color=(255 / 255, 127 / 255, 14 / 255), marker='o', label='scaled x_0_error_std')
    plt.legend(prop={'size': 16})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel('timesteps', size=16)
    plt.ylabel('std', size=16)
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()

    t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, 20, 30, 40, 50, 60, 70, 80, 90,
         100, 200, 300, 400, 500, 600, 700, 800, 900]

    x_0_error_std = [0.004999307711083627, 0.0068679020584265045, 0.008331700781733767, 0.009602449039145236,
                     0.010744002554323137, 0.011803540550621014, 0.012796417883691902, 0.013733058167720932,
                     0.014632054083449475, 0.015497966760449344,

                     0.016317762608802393, 0.02347621186466616,
                     0.029418241598856792, 0.03466819638500359, 0.0394669990625213, 0.04386737115783035,
                     0.04803230172061982, 0.05190638145601648, 0.05561026842406136,

                     0.05916737543642133,
                     0.08877058157425684, 0.11352599278082683, 0.13773979178949958, 0.16413153610968342,
                     0.19471968908813628, 0.2333616354056479, 0.285012498725943, 0.36540172226765816]


    scaled_x_0_error_std = [0.004999307711083627, 0.0036247510275675657, 0.003069618699765229, 0.0027607677916382154,
                            0.0025581786795807857, 0.0024144631776943817, 0.0023050679870451497, 0.00221746118047538,
                            0.0021462013799104633, 0.0020864547606752057,

                            0.002033066522154403, 0.0017226951184515897,
                            0.001555573287402555, 0.0014402505968195065, 0.0013531787515857294, 0.0012818682396865877,
                            0.0012238100988752484, 0.0011729727064434328, 0.0011294682176420994,

                            0.0010916298409949393,
                            0.0008637076552272943, 0.0007630232991053019, 0.0007204895659664609, 0.0007169047037418967,
                            0.0007454955722449934, 0.0008126106382598239, 0.00093037770449927, 0.0011467583264984811]



if __name__ == '__main__':
    pred_x_0_error_at_each_t()
    pred_x_0_error_accumulated()
