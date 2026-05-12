"""Parse DWD station list and produce dataset breakdown for project planning."""
import pandas as pd
from datetime import datetime

# Raw data from the uploaded file
raw = """00183 19810101 20260430             42     54.6791   13.4344 Arkona                                   Mecklenburg-Vorpommern
00656 19570101 20000731            607     51.7234   10.6021 Braunlage                                Niedersachsen
00662 19580101 20260430             81     52.2915   10.4464 Braunschweig                             Niedersachsen
00691 20090101 20260430              4     53.0451    8.7981 Bremen                                   Bremen
00853 19810101 20260430            416     50.7913   12.8720 Chemnitz                                 Sachsen
00867 20210101 20260430            344     50.3066   10.9679 Lautertal-Oberlauter                     Bayern
01048 19960901 20260430            228     51.1278   13.7543 Dresden-Klotzsche                        Sachsen
01303 20230831 20241231            150     51.4041    6.9677 Essen-Bredeney                           Nordrhein-Westfalen
01346 20230101 20260430           1486     47.8748    8.0038 Feldberg/Schwarzwald                     Baden-Wuerttemberg
01358 19580101 20260430           1213     50.4283   12.9536 Fichtelberg                              Sachsen
01420 20220508 20260430            100     50.0259    8.5213 Frankfurt/Main                           Hessen
01443 19730101 20150131            237     48.0232    7.8343 Freiburg                                 Baden-Wuerttemberg
01580 19760101 20131231            111     49.9859    7.9548 Geisenheim                               Hessen
01594 19740101 19901231             44     51.5068    7.0945 Gelsenkirchen                            Nordrhein-Westfalen
01639 20240101 20260430            203     50.6017    8.6439 Giessen/Wettenberg                       Hessen
01684 20010101 20260430            239     51.1621   14.9506 Goerlitz                                 Sachsen
01957 19810101 20150131             93     51.5138   11.9499 Halle-Kroellwitz                         Sachsen-Anhalt
01975 20050101 20260430             11     53.6332    9.9881 Hamburg-Fuhlsbuettel                     Hamburg
02290 19530101 20260430            977     47.8009   11.0108 Hohenpeissenberg                         Bayern
02483 20250101 20260430            839     51.1803    8.4891 Kahler Asten                             Nordrhein-Westfalen
02532 19790101 20021231            231     51.2963    9.4424 Kassel                                   Hessen
02712 19770101 20260430            428     47.6952    9.1307 Konstanz                                 Baden-Wuerttemberg
02925 20210101 20260430            356     51.3932   10.3123 Leinefelde                               Thueringen
02928 20061001 20190531            138     51.3151   12.4462 Leipzig-Holzhausen                       Sachsen
02932 20190501 20260430            131     51.4347   12.2396 Leipzig/Halle                            Sachsen
03015 19810101 20260430             98     52.2085   14.1180 Lindenberg                               Brandenburg
03028 19820101 20110831            157     51.7854    8.8388 Lippspringe-Bad                          Nordrhein-Westfalen
03032 19720101 20150131             25     55.0110    8.4125 List-auf-Sylt                            Schleswig-Holstein
03098 20060101 20130731            387     51.2452    7.6425 Luedenscheid                             Nordrhein-Westfalen
03478 19910101 20031231             81     53.5468   13.1914 Neubrandenburg                           Mecklenburg-Vorpommern
03631 19640101 20260430             12     53.7123    7.1519 Norderney                                Niedersachsen
03668 20090101 20260430            314     49.5030   11.0549 Nuernberg                                Bayern
03815 19770101 20011231             95     52.2553    8.0534 Osnabrueck                               Niedersachsen
03987 19451231 20260430             81     52.3812   13.0622 Potsdam                                  Brandenburg
04271 19980101 20260430              5     54.1803   12.0808 Rostock-Warnemuende                      Mecklenburg-Vorpommern
04336 19810101 20260430            319     49.2128    7.1077 Saarbruecken-Ensheim                     Saarland
04393 20080701 20260430              5     54.3279    8.6031 Sankt-Peter-Ording                       Schleswig-Holstein
04466 19800101 20260430             43     54.5275    9.5487 Schleswig                                Schleswig-Holstein
04642 19910101 20260430             22     52.8911   11.7297 Seehausen                                Sachsen-Anhalt
04928 19790101 20260430            314     48.8281    9.2000 Stuttgart-Schnarrenberg                  Baden-Wuerttemberg
05100 19580101 20260430            261     49.7479    6.6583 Trier-Petrisberg                         Rheinland-Pfalz
05142 20220101 20260430              1     53.7445   14.0698 Ueckermuende                             Mecklenburg-Vorpommern
05282 19500101 19960831            241     51.1188   13.6754 Wahnsdorf-bei-Dresden                    Sachsen
05404 19610101 20260430            477     48.4024   11.6946 Weihenstephan-Duernast                   Bayern
05419 19490101 20070430            264     50.9751   11.3076 Weimar                                   Thueringen
05440 19810101 20140930            439     49.0115   10.9308 Weissenburg-Emetzheim                    Bayern
05705 19570101 20260430            268     49.7704    9.9576 Wuerzburg                                Bayern
05779 19810101 20260430            877     50.7313   13.7516 Zinnwald-Georgenfeld                     Sachsen
05792 20130101 20260430           2956     47.4210   10.9848 Zugspitze                                Bayern
05856 19970102 20260430            476     48.5451   13.3532 Fuerstenzell                             Bayern
05906 19790101 20260430             98     49.5063    8.5584 Mannheim                                 Baden-Wuerttemberg
06197 20220101 20260430            258     51.8664    9.2710 Luegde-Paenbruch                         Nordrhein-Westfalen
07365 20090101 20260430            180     51.4458    7.2628 Bochum                                   Nordrhein-Westfalen
07370 20210101 20260430            499     49.3910   12.6838 Waldmuenchen                             Bayern
15000 20230101 20260430            231     50.7983    6.0244 Aachen-Orsbach                           Nordrhein-Westfalen
15444 20221029 20260331            593     48.4418    9.9216 Ulm-Maehringen                           Baden-Wuerttemberg"""

rows = []
for line in raw.strip().split("\n"):
    parts = line.split()
    sid = parts[0]
    von = datetime.strptime(parts[1], "%Y%m%d")
    bis = datetime.strptime(parts[2], "%Y%m%d")
    elev = int(parts[3])
    lat = float(parts[4])
    lon = float(parts[5])
    # Station name and Bundesland are the remainder
    rest = " ".join(parts[6:])
    # Last token is Bundesland; everything before is name
    tokens = rest.rsplit(maxsplit=1)
    name = tokens[0].strip() if len(tokens) > 1 else rest
    state = tokens[1] if len(tokens) > 1 else ""
    rows.append({
        "station_id": sid,
        "name": name,
        "state": state,
        "lat": lat,
        "lon": lon,
        "elev_m": elev,
        "von": von,
        "bis": bis,
        "years": (bis - von).days / 365.25,
    })

df = pd.DataFrame(rows)
print(f"Total stations in file: {len(df)}")
print()

# AlphaEarth Foundations: annual embeddings 2017-2025
ae_start = datetime(2017, 1, 1)
ae_end_recent = datetime(2025, 12, 31)
project_start = datetime(2020, 1, 1)  # we want 2020-2024 minimum

# Currently active = bis >= 2025
active = df[df["bis"] >= datetime(2025, 1, 1)]
print(f"Currently active stations (bis >= 2025-01-01): {len(active)}")

# Active AND covering the full AlphaEarth window 2017-2024
full_overlap = df[(df["von"] <= ae_start) & (df["bis"] >= datetime(2024, 12, 31))]
print(f"Stations covering full AlphaEarth window 2017-2024: {len(full_overlap)}")

# Active AND covering at least 2020-2024 (5 years AlphaEarth + recent obs)
core = df[(df["von"] <= project_start) & (df["bis"] >= datetime(2024, 12, 31))]
print(f"Stations covering 2020-2024 (our minimum target): {len(core)}")

# Active AND covering at least 2022-2024 (3 years; relaxed)
relaxed = df[(df["von"] <= datetime(2022, 1, 1)) & (df["bis"] >= datetime(2024, 12, 31))]
print(f"Stations covering 2022-2024 (relaxed; 3 years): {len(relaxed)}")

print()
print("=== Geographic spread of core 2020-2024 stations ===")
print(core.groupby("state").size().sort_values(ascending=False).to_string())
print()
print(f"Elevation range: {core['elev_m'].min()}m to {core['elev_m'].max()}m")
print(f"Latitude range: {core['lat'].min():.2f} to {core['lat'].max():.2f}")
print(f"Longitude range: {core['lon'].min():.2f} to {core['lon'].max():.2f}")

# Save the core list
core_export = core[["station_id", "name", "state", "lat", "lon", "elev_m", "von", "bis"]].copy()
core_export.to_csv("/home/claude/abhiram/dwd_core_stations.csv", index=False)
print()
print(f"Saved {len(core_export)} core stations to dwd_core_stations.csv")

# Also save the relaxed list (useful for ablations)
relaxed_export = relaxed[["station_id", "name", "state", "lat", "lon", "elev_m", "von", "bis"]].copy()
relaxed_export.to_csv("/home/claude/abhiram/dwd_relaxed_stations.csv", index=False)
print(f"Saved {len(relaxed_export)} relaxed stations to dwd_relaxed_stations.csv")
