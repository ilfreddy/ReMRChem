#!/bin/bash


./run_all_He_BS.sh; ./run_all_He_ABGV.sh
#./run_all_Ne_BS.sh; ./run_all_Ne_ABGV.sh;
#./run_all_Ar_BS.sh; ./run_all_Ar_ABGV.sh;
#./run_all_Hg_BS.sh; ./run_all_Hg_ABGV.sh



curl -X POST https://api.pushcut.io/MQFwODx_F6l1zq76_NHmN/notifications/notify_calc 