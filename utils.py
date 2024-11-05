import numpy as np
import pandas as pd
import torch
import os
import logging
from sklearn.preprocessing import MinMaxScaler


def preprecessingData(df):
    ## 데이터 전처리
    df['date'] = pd.to_datetime(df['날짜'].astype(str) + df['시간'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
    cols = list(df.columns)
    cols.remove('날짜')
    cols.remove('시간')
    cols.remove('date')
    df = df[["date"] + cols]
    df['date'] = pd.to_datetime(df['date'])

    df["거래대금"] = round(df["거래대금"]/1000000, 4)

    return df


## 추가 지표 만들기
# 볼린저 밴드 계산 함수
def calculatePriceBB(df, window=20, num_std=2):
    df['주가볼밴_중심선'] = df['종가'].rolling(window=window).mean()  # 20일 이동 평균
    df['STD20'] = df['종가'].rolling(window=window).std()   # 20일 표준 편차
    df['주가볼밴_상단선'] = df['주가볼밴_중심선'] + (num_std * df['STD20'])       # 상단 밴드
    df['주가볼밴_하단선'] = df['주가볼밴_중심선'] - (num_std * df['STD20'])       # 하단 밴드

    return df


# 이등분선 계산 함수
def calculate_yellow_box(df):

    most_high = df.loc[df.index[0], "고가"]
    most_low = df.loc[df.index[0], "저가"]
    yellow_line = []
    red_line = []

    for idx in df.index:

        if most_high < df.loc[idx, "고가"]:
            most_high = df.loc[idx, "고가"]

        if most_low > df.loc[idx, "저가"]:
            most_low = df.loc[idx, "저가"]

        avg = (most_high + most_low)/2
        yellow_line.append(avg)
        red_line.append((avg + most_high)/2)

    df["이등분선"] = yellow_line
    df["이등상단"] = red_line

    return df


# 거래대금 볼밴 그리기
def calculateTamountBB(df, window=20, num_std=2):
    df['거래대금_중심선'] = df['거래대금'].rolling(window=window).mean()  # 20일 이동 평균
    df_std = df['거래대금'].rolling(window=window).std()   # 20일 표준 편차
    df['거래대금_상단선'] = df['거래대금_중심선'] + (num_std * df_std)       # 상단 밴드
    df['거래대금_하단선'] = df['거래대금_중심선'] - (num_std * df_std)       # 하단 밴드
    df['거래대금_하단선'] = df['거래대금_하단선'].apply(lambda x: 0 if x < 0 else x)

    return df

## 라벨값 만들기
def makeLabel(df, rise_threshold=1.03, fall_threshold=0.985):
    df['Label'] = 0  # 기본값으로 0 설정
    num_rows = len(df)

    for i in range(num_rows):
        current_close = df.iloc[i]['종가']

        # 향후 시점의 가격 중에서 상승 및 하락 확인
        for j in range(i+1, num_rows):
            future_high = df.iloc[j]['고가']
            future_low = df.iloc[j]['저가']

            if future_high >= current_close * rise_threshold:
                df.iloc[i, df.columns.get_loc('Label')] = 1  # 3% 상승 시 1로 설정
                break
            elif future_low <= current_close * fall_threshold:
                df.iloc[i, df.columns.get_loc('Label')] = 0  # 1.5% 하락 시 0으로 설정
                break

    # Null 처리: 남은 행에서 아무 조건도 만족하지 않는 경우
    df['Label'].fillna(0, inplace=True)

    return df

# 슬라이딩 윈도우 함수
def sliding_window(data, window_size):
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i + window_size])
    return np.array(X)


def adjust_learning_rate(optimizer, scheduler, epoch, learning_rate, lradj, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == 'type3':
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif lradj == 'constant':
        lr_adjust = {epoch: learning_rate}
    elif lradj == '3':
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate*0.1}
    elif lradj == '4':
        lr_adjust = {epoch: learning_rate if epoch < 15 else learning_rate*0.1}
    elif lradj == '5':
        lr_adjust = {epoch: learning_rate if epoch < 25 else learning_rate*0.1}
    elif lradj == '6':
        lr_adjust = {epoch: learning_rate if epoch < 5 else learning_rate*0.1}
    elif lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + 'checkpoint.pth')
        self.val_loss_min = val_loss


# 로깅 설정
def setup_logger(log_dir='./logs', log_file='training_log.txt'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger('train_logger')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 로깅 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def makeDataset(LOAD, target="주도주", target_feature=['종가', '거래대금'], target_time="all"):

    if LOAD:
        joined_string = "_" + "_".join(target_feature)
        X_final = np.load(f'./dataset/{target_time}/_{target}_data_X{joined_string}.npy')
        y_final = np.load(f'./dataset/{target_time}/_{target}_data_y{joined_string}.npy')
    else:
        data_path = f'./{target}-주가데이터'

        all_X = []
        all_y = []
        window_size = 10

        for day_dir in os.listdir(data_path):
            day_path = f'{data_path}/{day_dir}'
            for filename in os.listdir(day_path):
                if filename.endswith(".csv"):
                    file_path = f'{day_path}/{filename}'
                    print("=" * 50)
                    print(file_path)

                    df = pd.read_csv(file_path, encoding='cp949').drop(columns=["누적체결매도수량", "누적체결매수수량"])

                    # 오류 처리. 날짜 겹친 파일은 row가 1200개 이상.
                    if df.shape[0] > 1200:
                        print("이상 발생")

                    else:
                        df = preprecessingData(df)

                        target_columns = ['종가', '거래대금', '시가', '고가', '저가']

                        if '거래량볼밴' in target_feature:
                            df = calculateTamountBB(df)
                            target_columns.extend(['거래대금_중심선', '거래대금_상단선', '거래대금_하단선'])
                        if '이등분선' in target_feature:
                            df = calculate_yellow_box(df)
                            target_columns.extend(['이등상단', '이등분선'])
                        if '주가볼밴' in target_feature:
                            df = calculatePriceBB(df)
                            target_columns.extend(['주가볼밴_중심선', '주가볼밴_상단선', '주가볼밴_하단선'])


                        # 타겟 날짜만 뽑기
                        df = df[df["date"] >= pd.to_datetime(f'2024-{day_dir[:2]}-{day_dir[2:]}')]
                        if target_time == "AM":
                          df = df[(df["date"].dt.time <= pd.to_datetime('13:00').time())]
                        df = df.reset_index(drop=True)

                        # 오류 처리. 데이터가 없는 경우 있음
                        if df.empty == False:

                            ## 라벨 만들기
                            df = makeLabel(df)

                            ## 학습 데이터셋 만들기
                            feature_columns = df[target_columns].values
                            label_column = df['Label'].values
                            scaler = MinMaxScaler()
                            feature_columns_scaled = scaler.fit_transform(feature_columns)

                            # df_stamp = df[['date']]
                            # df_stamp['date'] = pd.to_datetime(df_stamp.date)
                            # data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='t').transpose()
                            # feature_columns_scaled = np.concatenate((feature_columns_scaled, data_stamp), axis=1)
                            # print(feature_columns_scaled.shape)
                            X_windows = sliding_window(feature_columns_scaled, window_size)
                            # print(X_windows.shape)
                            y_windows = label_column[window_size - 1:]

                            if X_windows.size > 0 and y_windows.size > 0:
                                all_X.append(X_windows)
                                all_y.append(y_windows)
                            else:
                                print("Skipped empty processed data")




        X_final = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        print(X_final.shape)
        print(y_final.shape)

        joined_string = "_" + "_".join(target_feature)
        np.save(f'./dataset/{target_time}/_{target}_data_X{joined_string}.npy', X_final)
        np.save(f'./dataset/{target_time}/_{target}_data_y{joined_string}.npy', y_final)

    return X_final, y_final