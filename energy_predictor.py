import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy as sp
from scipy.stats import shapiro

logger_ = logging.getLogger(__name__)
logger_.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s]-[%(name)s]-[%(levelname)s]: %(message)s')
handler.setFormatter(formatter)
logger_.addHandler(handler)


class EnergyPredictionDataset(object):

    @staticmethod
    def read_energy_tm_data_(src_file_name):
        data_frame = pd.read_csv(src_file_name)
        return data_frame

    @staticmethod
    def clean_and_validate_(src_file_name):

        """
        Parsing energy demand file given. The content of the form:
        <dy>,<station_01_temperature_degrees_celcius>,<station_02_temperature_degrees_celcius>,
          <station_01_wind_speed_m_per_s>,<station_02_wind_speed_m_per_s>,<gas_demand_mcm>

        :param src_file_name: source file given as part of exercise
        :return: Pandas data frame containing file data
        """
        try:
            df = EnergyPredictionDataset.read_energy_tm_data_(src_file_name)
            df.dropna()
            df['dy'] = pd.to_datetime(df['dy'], format='%d/%m/%Y')
            df.columns = ['date',
                          'station_01_temperature',
                          'station_02_temperature',
                          'station_01_wind',
                          'station_02_wind',
                          'gas_demand_volume']
            df = df[pd.to_numeric(df['gas_demand_volume'], errors='coerce').notnull()]
            df = df[pd.to_numeric(df['station_01_temperature'], errors='coerce').notnull()]
            with pd.ExcelWriter('weather_and_gas_demand_historic_actuals_cleaned.xlsx') as writer:
                df.to_excel(writer, sheet_name='Cleaned-Data')

            return df

        except Exception as ex:
            logger_.error("Failed. Msg: {}".format(ex))
            return None

    @staticmethod
    def build(src_file_name):
        return EnergyPredictionDataset.clean_and_validate_(src_file_name)


def main(argv):

    logger_.info("Starting program .. ")
    # create correlation matrix of all energy demand features
    data_frame = EnergyPredictionDataset.build("weather_and_gas_demand_historic_actuals.csv")
    if data_frame is None:
        logger_.error("Unexpected exception while building data frame.")
        return 1

    # Plot dataset
    data_frame.plot(x='station_01_temperature', y='gas_demand_volume', style='o')
    plt.title('Station 01 Temperature vs Gas Demand')
    plt.xlabel('Station_01_Temperature')
    plt.ylabel('Gas_Demand')
    plt.show()
    plt.close()

    description = data_frame.describe()
    with pd.ExcelWriter('description.xlsx') as writer:
        description.to_excel(writer, sheet_name='Info')

    # Build correlation matrix
    data_frame_no_date = data_frame.drop(['date'], axis=1, inplace=False)
    corr_matrix_full = data_frame_no_date.corr()
    print(corr_matrix_full)
    with pd.ExcelWriter('correlation_matrix_full.xlsx') as writer:
        corr_matrix_full.to_excel(writer, sheet_name='CORR-VER-1')

    # remove unnecessary columns
    corr_matrix_stripped = corr_matrix_full.drop(['station_02_temperature',
                                                  'station_02_wind'],
                                                 axis=1, inplace=False)
    # remove unnecessary rows
    corr_matrix_stripped.drop(['station_02_temperature', 'station_02_wind'], inplace=True)
    with pd.ExcelWriter('correlation_matrix_stripped.xlsx') as writer:
        corr_matrix_stripped.to_excel(writer, sheet_name='CORR-VER-2')

    # ------------------------------------------------------------------------------------
    # stripped correlation matrix of the station 2 parameters as correlated
    # ------------------------------------------------------------------------------------
    #                          station_01_temperature  station_01_wind	   gas_demand
    #  station_01_temperature  1	                   -0.205899783	       -0.940276696
    #  station_01_wind	       -0.205899783	           1	               0.257729078
    #  gas_demand	           -0.940276696	           0.257729078	       1
    # ------------------------------------------------------------------------------------

    # Now remove station_01_wind column and row from correlation matrix
    corr_matrix = corr_matrix_stripped.drop(['station_01_wind'], inplace=False)
    corr_matrix.drop(['station_01_wind'], axis=1, inplace=True)
    with pd.ExcelWriter('correlation_matrix.xlsx') as writer:
        corr_matrix.to_excel(writer, sheet_name='CORR-MTX-FINAL')

    # Perform test for normality of distribution of station_01_temperature values
    distance, p_value = sp.stats.normaltest(data_frame['station_01_temperature'])
    stat, s_p_value = shapiro(data_frame['station_01_temperature'])

    # Split data into training and testing sets
    x = data_frame['station_01_temperature'].values.reshape(-1,1)
    y = data_frame['gas_demand_volume'].values.reshape(-1,1)
    dates = data_frame['date'].values.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2105,
                                                        shuffle=False)
    x_train_, x_test_, dates_train, dates_test = train_test_split(x, dates,
                                                                  test_size=0.2105,
                                                                  shuffle=False)

    # Build linear regression model - single predictor and single target
    regressor = LinearRegression()

    # Train the algorithm
    regressor.fit(x_train, y_train)

    # Predict gas demand for July 2018
    y_pred = regressor.predict(x_test)

    # Plot
    plt.clf()
    plt.scatter(x_test, y_test,  color='gray')
    plt.plot(x_test, y_pred, color='red', linewidth=2)
    plt.show()

    # Compute common erros for timeseries datasets
    logger_.info("Mean Absolute Error: {}".format(metrics.mean_absolute_error(y_test, y_pred)))
    logger_.info("Mean Squared Error: {}".format(metrics.mean_squared_error(y_test, y_pred)))
    logger_.info("Root Mean Squared Error: {}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

    # Create output and write to external file
    new_df = pd.DataFrame({'Date': dates_test.flatten(),
                           'Actual Demand': y_test.flatten(),
                           'Predicted Demand': y_pred.flatten()})
    with pd.ExcelWriter('new_df.xlsx') as writer:
        new_df.to_excel(writer, sheet_name='Info')
    logger_.info("Completed program execution.. ")
    return 0


if __name__ == '__main__':
    status = main(sys.argv[1:])
    sys.exit(status)

