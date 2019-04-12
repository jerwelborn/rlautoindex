#!/bin/bash

# run within something like a tmux session in background

# run from rlautoindex/root
if [ "$(basename $(pwd))" != "rlautoindex" ]
then 
    echo "start script from rlautoindex root" ;
    exit ;
fi

# base directory for workload, workload results with timestamp
RESULT_DIR=../res/ 
RESULT_DIR=${RESULT_DIR}$(date '+%m-%d-%y_%H:%M')
mkdir ${RESULT_DIR}

cp conf/{dqn,spg,experiment}.json ${RESULT_DIR}

#
# build workload, run default
# 
echo "#### RUNNING DEFAULT ON WORKLOAD ####"
time python3 src/common/postgres_controller.py \
--dqn=True \
--config=conf/dqn.json \
--experiment_config=conf/experiment.json \
--result_dir=${RESULT_DIR} \
--generate_workload=True \
--with_agent=False &&

#
# run dqn
#
echo "#### RUNNING DQN ON WORKLOAD ####" &&
time python3 src/common/postgres_controller.py \
--dqn=True \
--config=conf/dqn.json \
--experiment_config=conf/experiment.json \
--result_dir=${RESULT_DIR} \
--generate_workload=False 
&> ${RESULT_DIR}/dqn.log &&

mv ${RESULT_DIR}/dqn.log ${RESULT_DIR}/dqn &&

#
# run spg
#
echo "#### RUNNING SPG ON WORKLOAD ####" &&
time python3 src/common/postgres_controller.py \
--dqn=False \
--config=conf/spg.json \
--experiment_config=conf/experiment.json \
--result_dir=${RESULT_DIR} \
--generate_workload=False \
&> ${RESULT_DIR}/spg.log &&

mv ${RESULT_DIR}/spg.log ${RESULT_DIR}/spg &&

#
# run tuner
#
echo "#### RUNNING TUNER ON WORKLOAD ####" &&
time python src/baseline/postgres_tuner.py \
--experiment_config=conf/experiment.json \
--data_dir=${RESULT_DIR} \
&> ${RESULT_DIR}/tuner.log &&

mv ${RESULT_DIR}/tuner.log ${RESULT_DIR}/tuner