import numpy as np
from scipy.optimize import minimize
from model import Learner_SuEIrD
from data import *
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from rolling_train_new import *
import argparse

parser = argparse.ArgumentParser(description='training omicron for different regions')
parser.add_argument('--region', default = 'US', help='region')

args = parser.parse_args()

if __name__ == '__main__':


    # N = 100000000
    # E = N/400

    if args.region == "US":
        N = 65853700
        
        # rs = np.asarray([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
        rs = np.asarray([100, 105, 110, 115, 120, 140, 145, 150, 155, 160, 170])

        data = JHU_global()
        # data = NYTimes(level='states')

        start_date = '2021-11-23'

        # train_data_raw = [data.get(start_date, '2021-12-16', "US"), data.get('2021-12-16', '2022-1-27', "US")]
        data_raw = list(data.get(start_date, '2022-4-10', "US"))

        temp = list(data.get('2021-11-15', '2021-11-16', 'US'))

        data = []
        
        data_confirm = data_raw[0] - temp[0][0]
        data_fatality = data_raw[1] - temp[1][0]

        data += [rolling_average(data_confirm), rolling_average(data_fatality)]

        train_data = [[data[0][:27], data[1][:27]], [data[0][27:62], data[1][27:62]]]
        # train_data = [[data[0][:55], data[1][:55]]]
        data_confirm = train_data[0][0]
        data_fatality = train_data[0][1]

        val_data = [data[0][61:87], data[1][61:87]]
        test_data = [data[0][87:], data[1][87:]]

    elif args.region == 'California':
        N = 12300000
        
        # rs = np.asarray([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
        rs = np.asarray([100, 105, 110, 115, 120, 140, 145, 150, 155, 160, 170])

        data = JHU_US(level='states')

        start_date = '2021-11-30'

        # train_data_raw = [data.get(start_date, '2021-12-16', "US"), data.get('2021-12-16', '2022-1-27', "US")]
        data_raw = list(data.get(start_date, '2022-4-10', "California"))

        temp = list(data.get('2021-11-15', '2021-11-16', 'California'))

        data = []
        
        data_confirm = data_raw[0] - temp[0][0]
        data_fatality = data_raw[1] - temp[1][0]

        data += [rolling_average(data_confirm), rolling_average(data_fatality)]

        train_data = [[data[0][:30], data[1][:30]], [data[0][30:57], data[1][30:57]]]
        # train_data = [[data[0][:55], data[1][:55]]]
        data_confirm = train_data[0][0]
        data_fatality = train_data[0][1]

        val_data = [data[0][57:87], data[1][57:87]]
        test_data = [data[0][87:], data[1][87:]]

        # plt.xticks([])
        # plt.yticks([])
        # plt.plot(np.diff(train_data[0][0].tolist()))
        # plt.plot(np.diff(train_data[1][0].tolist()))
        # plt.title("Averaged data")
        # # plt.plot(train_data[1][0])
        # plt.show()

    train_start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date() + dt.timedelta(days=7)
    train_end_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date() + dt.timedelta(days=len(train_data[0][0]) + len(train_data[1][0]))
    val_end_date = train_end_date + dt.timedelta(days=len(val_data[0]))
    test_end_date = val_end_date + dt.timedelta(days=len(test_data[0]))

    print(f"Train data range: {train_start_date} to {train_end_date}")
    print(f"Val data range: {train_end_date} to {val_end_date}")
    print(f"Test data range: {val_end_date} to {test_end_date}")

    # plt.figure()
    # plt.plot(np.diff(train_data[0][0].tolist() + train_data[1][0].tolist()))
    # # plt.plot(train_data[1][0])
    # plt.show()

    daily_increases = []

    pred_confirms, pred_fatalities = [], []

    val_log = []
    min_val_loss = 1e9

    valid_rs = []

    for r in rs:
        E = N / r

        init = [N-E-data_confirm[0]-data_fatality[0],
                E, data_confirm[0], data_fatality[0]]


        model = Learner_SuEIrD(N=N, E_0=E, I_0=data_confirm[0], D_0=data_fatality[0])

        params_all, loss_all = rolling_train(model, init, train_data)
        
        val_loss = validation(model, init, params_all, train_data, val_data)

        for params in params_all:
            beta, gamma, sigma, mu, omega = params
            # we cannot allow mu>sigma otherwise the model is not valid
            if mu > sigma or omega > gamma:
                val_loss = 1e6    

        pred_confirm, pred_fatality, _ = rolling_prediction(model, init, params_all, train_data, pred_range=60, daily_smooth=True)

        max_daily_confirm = np.max(np.diff(pred_confirm))
        pred_confirm_last, pred_fatality_last = pred_confirm[-1], pred_fatality[-1]
        # print(np.diff(pred_fatality))
        # print(sigma/mu)
        #prevent the model from explosion
        # if pred_confirm_last >  8*train_data[-1][0][-1] or  np.diff(pred_confirm)[-1]>=np.diff(pred_confirm)[-2]:
        #     val_loss = 1e8

        print(f"r = {r}, val_loss = {val_loss}")
        if val_loss < min_val_loss:
            # record the information for validation
            min_val_loss = val_loss
            optimal_r = r
            optimal_params = [init, params_all]
            valid_rs.append(r)

            val_log += [[N, E] + [val_loss] + [pred_confirm_last] + [pred_fatality_last] + [max_daily_confirm] + loss_all  ]

            pred_confirms.append(pred_confirm)
            pred_fatalities.append(pred_fatality)

            confirm = train_data[0][0].tolist() + train_data[1][0].tolist() + pred_confirm[1:].tolist()
            # confirm = train_data[0][0].tolist() + pred_confirm[1:-1].tolist()

            # daily_increases.append(rolling_average(np.diff(np.array(confirm))))
            daily_increases.append(np.diff(np.array(confirm)))
        

    # print (np.asarray(val_log))
    # best_log = np.array(val_log)[np.argmin(np.array(val_log)[:,2]),:]
    # print("Best Val loss: ", best_log[2], " Last CC: ", best_log[3], " Last FC: ", best_log[4], " Max inc Confirm: ", best_log[5] )

    test_loss = validation(model, optimal_params[0], optimal_params[1], train_data, test_data)
    print(f"Optimal r = {optimal_r}, best Val loss: {min_val_loss}, Test loss: {test_loss}")
    
    beta, gamma, sigma, mu, omega = optimal_params[1][0]
    print(f"Optimal model parameters for the first part: beta = {beta}, gamma = {gamma}, sigma = {sigma}, mu = {mu}, omega = {omega}")

    beta, gamma, sigma, mu, omega = optimal_params[1][1]
    print(f"Optimal model parameters for the second part: beta = {beta}, gamma = {gamma}, sigma = {sigma}, mu = {mu}, omega = {omega}")

    start = train_start_date
    end = start + dt.timedelta(days=len(daily_increases[0]))
    days = mdates.drange(start, end, dt.timedelta(days=1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.tick_params(axis='x', labelrotation=30)

    ax.set_title(f'Daily new cases for {args.region}')

    i = 0
    for daily_increase in daily_increases:
        ax.plot(days, daily_increase, label=f"r={valid_rs[i]}")
        i += 1


    # true_data = rolling_average(np.diff(data.get(start_date, '2022-04-10', 'US')[0]))
    # true_data = np.diff(data.get(start_date, '2022-04-10', 'US')[0])
    ax.plot(days, np.diff(data[0][:len(days)+1]), label='ground truth')

    ax.legend()
    plt.show()
    plt.close()