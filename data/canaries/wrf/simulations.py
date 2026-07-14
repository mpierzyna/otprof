"""Simulation definition for a single 5-day WRF run over the Canary Islands."""

from wrf_massive.base import BBox, Simulation

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

# Lat/lon box the CDS forcing download must cover. Fully contains the 280x150 @ 2km Canaries domain
# (centre 28.4N/15.7W) with margin; validated against namelist.tmpl.wps by PullCdsStage.
canaries_area = BBox(north=31.0, west=-20.5, south=25.5, east=-11.0)

# Single 5-day net simulation (summer trade-wind regime) with a 12h spin-up prepended.
sim_canaries = Simulation(
    begin="2020-07-01T00:00:00",
    end="2020-07-06T00:00:00",
    warmup_h=12,
    sim_dir="sim_2020-07-01",
    settings=mynn25,
    area=canaries_area,
)


if __name__ == "__main__":
    print(sim_canaries)
