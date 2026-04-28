from typing import List
import datetime
import numpy as np

from wrf_massive.base import Simulation

mynn25 = {
    # MYNN 2.5
    "physics__sf_sfclay_physics": "5",
    "physics__bl_pbl_physics": "5",
    "physics__bl_mynn_closure": "2.5",
    # without EDMF (so, mixing length 1)
    "physics__bl_mynn_mixlength": "1",
    "physics__bl_mynn_edmf": "0",
    "physics__bl_mynn_edmf_mom": "0",
    "physics__bl_mynn_edmf_tke": "0",
    "physics__sf_urban_physics": "0",  # no urban model
}


def get_1y_sims() -> List[Simulation]:
    n_days = 5  # net length of sim in days
    begin_days = np.arange(0, 365, n_days)
    # print(begin_days, len(begin_days))

    begin_years = [2017, 2018, 2019, 2020]
    begin_years = np.tile(begin_years, np.ceil(len(begin_days) / len(begin_years)).astype(int))[: len(begin_days)]
    # print(begin_years, len(begin_years))

    sims = []
    for d, y in zip(begin_days, begin_years):
        begin = datetime.datetime(y, 1, 1) + datetime.timedelta(days=int(d))
        end = begin + datetime.timedelta(days=n_days)
        sim = Simulation(
            begin=begin,
            end=end,
            warmup_h=12,
            sim_dir=f"sim_{begin.strftime('%Y-%m-%d')}",
            settings=mynn25,
        )
        sims.append(sim)
    return sims


sims_1y = get_1y_sims()
sims_1y_dev = sims_1y[::10]  # every 10th sim for dev
sim_dev = Simulation(
    begin="2020-01-01T00:00:00",
    end="2020-01-02T00:00:00",
    warmup_h=12,
    sim_dir="test",
    settings=mynn25,
)


if __name__ == "__main__":
    sims = get_1y_sims()[::10]
    for s in sims:
        print(s)
