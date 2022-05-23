import numpy as np
from scipy.optimize import minimize
from model import Learner_SuEIrD
from data import JHU_global
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


def loss(pred, target, smoothing=10): 
    return np.mean((np.log(pred+smoothing) - np.log(target+smoothing))**2)

def train(model, init, prev_params, train_data, reg=0):
    # prev_params is not used, this is for the usage of defining regularization loss

    data_confirm, data_fatality = train_data[0], train_data[1]
    # if len(train_data)==3:
    #     data_fatality = train_data[1] + train_data[2]
    size = len(data_confirm)

    fatality_perday = np.diff(data_fatality)
    target_ave_fatality_perday = np.median(fatality_perday[np.maximum(0, len(fatality_perday)-7):])
    confirm_perday = np.diff(data_confirm)
    target_ave_confirm_perday = np.median(confirm_perday[np.maximum(0, len(confirm_perday)-7):])

    def loss_train(params): # loss function for training

        _, _, _, pred_fatality, _, pred_confirm = model(size, params, init)

        pred_fatality = pred_fatality + data_fatality[0] - pred_fatality[0]
        reg = 0.5
        # if len(train_data)==3: # the data includes recovered cases
        #     pred_fatality = pred_remove
        #     reg = 0
        pred_ave_confirm_perday = np.mean(np.maximum(0, np.diff(pred_confirm)[-7:]))
        pred_ave_fatality_perday = np.mean(np.maximum(0, np.diff(pred_fatality)[-7:]))

        # Use both daily cases and cumulative cases to construct the loss function
        return loss(pred_confirm, data_confirm) + loss(pred_fatality, data_fatality) + \
                1*loss(pred_ave_confirm_perday, target_ave_confirm_perday) + 3 * \
                loss(pred_ave_fatality_perday, target_ave_fatality_perday) 

    # scipy optimizer
    optimal = minimize(
        loss_train,
        [0.2, .5e-2, 2.5e-1, 0.01, 0.00001],
        method='L-BFGS-B',
        bounds=[(0.0001, .3), (0.001, 0.3), (0.01, 1), (0.001, 1.), (0.000001, 0.005)]
    )

    return optimal.x, optimal.fun


def rolling_train(model, init, train_data):

    # train multiple models in a rolling manner, the susceptible and exposed populations will be transfered to the next period as initialization


    lag = 0
    params_all = []
    loss_all = []
    prev_params = [0.2, .5e-2, 3e-1, 0.01, 0.0001]
    reg = 0
    model.reset()
    N = model.N
    ind = 0

    for _train_data in train_data:
        ind += 1
        data_confirm, data_fatality = _train_data[0], _train_data[1]
        params, train_loss = train(model, init, prev_params, _train_data, reg=reg)
        pred_S, pred_E, pred_I, pred_D, pred_r, pred_confirm = model(len(data_confirm), params, init)

        pred_remove = np.add(pred_r, pred_D)

        lag += len(data_confirm)-10
        reg = 0

        # if len(_train_data)==3:
        #     true_remove = np.minimum(data_confirm[-1], np.maximum(_train_data[1][-1] + _train_data[2][-1], pred_remove[-1]))
        # else:
        true_remove = np.minimum(data_confirm[-1], pred_remove[-1])
        # true_remove = pred_remove[-1]

        init = [pred_S[-1], pred_E[-1], data_confirm[-1]-true_remove, true_remove]
        
        prev_params = params
        params_all += [params]
        loss_all += [train_loss]
    
    # init[0] = init[0] - new_sus
    model.reset()
    # pred_S, pred_E, pred_I, pred_D, pred_r, pred_confirm = model(7, params, init, lag=lag)
    
    
    return params_all, loss_all 

def rolling_prediction(model, init, params_all, train_data, pred_range, daily_smooth=False):
    model.reset()
    ind = 0
    for _train_data, params in zip(train_data, params_all):

        ind += 1
        data_confirm, data_fatality = _train_data[0], _train_data[1]
        pred_sus, pred_exp, pred_act, pred_fatality, pred_recover, pred_confirm = model(len(data_confirm), params, init)
        
        pred_remove = np.add(pred_fatality, pred_recover)
        
        true_remove = np.minimum(data_confirm[-1], pred_remove[-1])
        # true_remove = pred_remove[-1]

        init = [pred_sus[-1], pred_exp[-1], data_confirm[-1]-true_remove, true_remove]

    pred_sus, pred_exp, pred_act, pred_fatality, pred_recover, pred_confirm = model(pred_range, params, init)
    pred_fatality = pred_fatality + train_data[-1][1][-1] - pred_fatality[0]

    # using average results to smoothing the fatality and confirmed case predictions
    fatality_perday = np.diff(np.asarray(data_fatality))
    ave_fatality_perday = np.mean(fatality_perday[-7:])

    confirm_perday = np.diff(np.asarray(data_confirm))
    ave_confirm_perday = np.mean(confirm_perday[-7:])

    slope_fatality_perday  = np.mean(fatality_perday[-7:] -fatality_perday[-14:-7] )/7
    slope_confirm_perday  = np.mean(confirm_perday[-7:] -confirm_perday[-14:-7] )/7

    smoothing = 1. if daily_smooth else 0



    temp_C_perday = np.diff(pred_confirm.copy())
    slope_temp_C_perday = np.diff(temp_C_perday)
    modified_slope_gap_confirm = (slope_confirm_perday - slope_temp_C_perday[0])*smoothing

    modified_slope_gap_confirm = np.maximum(np.minimum(modified_slope_gap_confirm, ave_confirm_perday/40), -ave_confirm_perday/100)
    slope_temp_C_perday = [slope_temp_C_perday[i] + modified_slope_gap_confirm * np.exp(-0.05*i**2) for i in range(len(slope_temp_C_perday))]
    temp_C_perday = [np.maximum(0, temp_C_perday[0] + np.sum(slope_temp_C_perday[0:i])) for i in range(len(slope_temp_C_perday)+1)]

    # modifying_gap_confirm = (ave_confirm_perday - temp_C_perday[0])*smoothing
    # temp_C_perday  = [np.maximum(0, temp_C_perday[i] + modifying_gap_confirm * np.exp(-0.1*i)) for i in range(len(temp_C_perday))]
    temp_C =  [pred_confirm[0] + np.sum(temp_C_perday[0:i])  for i in range(len(temp_C_perday)+1)]
    pred_confirm = np.array(temp_C)



    temp_F_perday = np.diff(pred_fatality.copy())
    slope_temp_F_perday = np.diff(temp_F_perday)
    smoothing_slope = 0 if np.max(fatality_perday[-7:])>4*np.median(fatality_perday[-7:]) or np.median(fatality_perday[-7:])<0 else 1
    
    modified_slope_gap_fatality = (slope_fatality_perday - slope_temp_F_perday[0])*smoothing_slope
    modified_slope_gap_fatality = np.maximum(np.minimum(modified_slope_gap_fatality, ave_fatality_perday/10), -ave_fatality_perday/20)
    slope_temp_F_perday = [slope_temp_F_perday[i] + modified_slope_gap_fatality * np.exp(-0.05*i**2) for i in range(len(slope_temp_F_perday))]
    temp_F_perday = [np.maximum(0, temp_F_perday[0] + np.sum(slope_temp_F_perday[0:i])) for i in range(len(slope_temp_F_perday)+1)]


    modifying_gap_fatality = (ave_fatality_perday - temp_F_perday[0])*smoothing_slope
    temp_F_perday  = [np.maximum(0, temp_F_perday[i] + modifying_gap_fatality * np.exp(-0.05*i)) for i in range(len(temp_F_perday))]
    temp_F =  [pred_fatality[0] + np.sum(temp_F_perday[0:i])  for i in range(len(temp_F_perday)+1)]
    pred_fatality = np.array(temp_F)

    model.reset()
    return pred_confirm, pred_fatality, pred_act

def rolling_average(arr):
    window_size = 7
  
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    
    # Loop through the array t o
    #consider every window of size 3
    while i < len(arr) - window_size + 1:
    
        # Calculate the average of current window
        window_average = int(round(np.sum(arr[i:i+window_size]) / window_size, 0))
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1

    return np.array(moving_averages)

def validation(model, init, params_all, train_data, val_data):
    val_data_confirm, val_data_fatality = val_data[0], val_data[1]
    val_size = len(val_data_confirm)

    pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, pred_range=val_size)
    return  0.5*loss(pred_confirm, val_data_confirm, smoothing=0.1) + loss(pred_fatality, val_data_fatality, smoothing=0.1)

if __name__ == '__main__':


    # N = 100000000
    # E = N/400

    N = 65853700*3
    
    rs = np.asarray([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
    # rs = np.asarray([100, 105, 110, 115, 120, 140, 145, 150, 155, 160, 170])

    data = JHU_global()
    # data = NYTimes(level='states')
    a, decay = 0.1, 0.05
    # state = "California"

    start_date = '2021-6-12'

    # train_data_raw = [data.get(start_date, '2021-12-16', "US"), data.get('2021-12-16', '2022-1-27', "US")]
    data_raw = list(data.get(start_date, '2022-4-10', "US"))

    temp = list(data.get('2021-6-1', '2021-6-2', 'US'))

    data = []
    
    data_confirm = data_raw[0] - temp[0][0]
    data_fatality = data_raw[1] - temp[1][0]

    data += [rolling_average(data_confirm), rolling_average(data_fatality)]

    train_data = [[data[0][:156], data[1][:156]], [data[0][156:201], data[1][156:201]]]
    # train_data = [[data[0][:55], data[1][:55]]]
    data_confirm = train_data[0][0]
    data_fatality = train_data[0][1]

    val_data = [data[0][201:271], data[1][201:271]]
    # print(len(val_data[0]))

    # tt = np.diff(train_data[0][0].tolist() + train_data[1][0].tolist())

    plt.figure()


    # plt.subplot(1, 2, 1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.plot(np.diff(data_raw[0].tolist())[:len(tt)])
    # plt.title("Raw data")

    # plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.plot(np.diff(train_data[0][0].tolist()))
    plt.plot(np.diff(train_data[1][0].tolist()))
    plt.title("Averaged data")
    # plt.plot(train_data[1][0])
    plt.show()

    daily_increases = []
    daily_fatality_increases = []

    pred_confirms, pred_fatalities = [], []

    val_log = []
    min_val_loss = 1e9

    for r in rs:
        E = N / r

        init = [N-E-data_confirm[0]-data_fatality[0],
                E, data_confirm[0], data_fatality[0]]


        model = Learner_SuEIrD(N=N, E_0=E, I_0=data_confirm[0], D_0=data_fatality[0])

        params_all, loss_all = rolling_train(model, init, train_data)
        
        val_loss = validation(model, init, params_all, train_data, val_data)
        print(val_loss)

        for params in params_all:
            beta, gamma, sigma, mu, omega = params
            print(f"beta = {beta}, gamma = {gamma}, sigma = {sigma}, mu = {mu/sigma}, omega = {omega/gamma}")
            # we cannot allow mu>sigma otherwise the model is not valid
            if mu > sigma or omega > gamma:
                val_loss = 1e6    

        pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, pred_range=80, daily_smooth=True)

        max_daily_confirm = np.max(np.diff(pred_confirm))
        pred_confirm_last, pred_fatality_last = pred_confirm[-1], pred_fatality[-1]
        # print(np.diff(pred_fatality))
        # print(sigma/mu)
        #prevent the model from explosion
        if pred_confirm_last >  8*train_data[-1][0][-1] or  np.diff(pred_confirm)[-1]>=np.diff(pred_confirm)[-2]:
            val_loss = 1e6

        print(f"r = {r}, val_loss = {val_loss}")
        if val_loss < min_val_loss:
        # record the information for validation
            val_log += [[N, E] + [val_loss] + [pred_confirm_last] + [pred_fatality_last] + [max_daily_confirm] + loss_all  ]

            pred_confirms.append(pred_confirm)
            pred_fatalities.append(pred_fatality)

            confirm = train_data[0][0].tolist() + train_data[1][0].tolist() + pred_confirm[1:].tolist()
            fatality = train_data[0][1].tolist() + train_data[1][1].tolist() + pred_fatality[1:].tolist()
            # confirm = train_data[0][0].tolist() + pred_confirm[1:-1].tolist()

            # daily_increases.append(rolling_average(np.diff(np.array(confirm))))
            daily_increases.append(np.diff(np.array(confirm)))
            daily_fatality_increases.append(np.diff(np.array(fatality)))

        min_val_loss = np.minimum(val_loss, min_val_loss)

    print (np.asarray(val_log))
    best_log = np.array(val_log)[np.argmin(np.array(val_log)[:,2]),:]
    print("Best Val loss: ", best_log[2], " Last CC: ", best_log[3], " Last FC: ", best_log[4], " Max inc Confirm: ", best_log[5] )

    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = start + dt.timedelta(days=len(daily_increases[0]))
    days = mdates.drange(start, end, dt.timedelta(days=1))

    fig = plt.figure()
    ax = fig.add_subplot()
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.tick_params(axis='x', labelrotation=45)

    i = 0
    for daily_increase in daily_increases:
        ax.plot(days, daily_increase, label=rs[i])
        i += 1


    # true_data = rolling_average(np.diff(data.get(start_date, '2022-04-10', 'US')[0]))
    # true_data = np.diff(data.get(start_date, '2022-04-10', 'US')[0])
    ax.plot(days, np.diff(data[0][:len(days)+1]), label='actual data')

    ax.legend()
    plt.show()

    fig = plt.figure()

    ax = fig.add_subplot()
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.tick_params(axis='x', labelrotation=45)

    i = 0
    for daily_increase in daily_fatality_increases:
        ax.plot(days, daily_increase, label=rs[i])
        i += 1


    # true_data = rolling_average(np.diff(data.get(start_date, '2022-04-10', 'US')[0]))
    # true_data = np.diff(data.get(start_date, '2022-04-10', 'US')[0])
    ax.plot(days, np.diff(data[1][:len(days)+1]), label='actual data')

    ax.legend()
    plt.show()

    plt.close()