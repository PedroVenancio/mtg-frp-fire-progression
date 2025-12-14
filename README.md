# MTG FRP Fire Progression Analyzer with Intensity Calculation

Python script for create hourly and cumulative fire progression polygons from MTG (Meteosat Third Generation) FRP (Fire Radiative Power) data, with calibration against reference burned areas and fire intensity estimation.

https://github.com/user-attachments/assets/198a031a-720d-4a23-9ee9-2cea6e173c95

   - **Fire Intensity based on Radiative Method**: I_rad = FRP / (L × X_r)

![Fire_Intensity](https://github.com/user-attachments/assets/f30c0364-130b-40cd-a345-0fc484331702)

   - **Fire Intensity based on Traditional Method**: I_trad = H × w × v

![Fire_Intensity_Byram](https://github.com/user-attachments/assets/00306d45-7eb2-4f1d-bfd5-cca42bc53648)


## 📋 Description

This script processes MTFRPPixel fire detection data to create temporal progression polygons of wildfires. It supports both cumulative and hourly progression modes, with advanced filtering, calibration capabilities, and now includes **fire intensity estimation** using radiative fraction approach.

## ✨ Features

- **Dual Processing Modes**: Cumulative progression and hourly progression for analysis and animations
- **Non-Overlapping Progression**: Option for print layouts showing only new burned areas each hour
- **Advanced Filtering**: FRP thresholding, spatial density filtering (DBSCAN), cluster size filtering and minimum area filtering
- **Flexible Hull Generation**: Concave hull with fallback to convex hull
- **Calibration System**: Automatic parameter calibration using reference burned areas
- **Temporal Continuity**: Fill missing hours to maintain temporal (hourly) progression
- **Area Correction**: Negative buffer (shrink) to mitigate area overestimation
- **Multiple Outputs**: Cumulative and hourly polygons, daily summaries, calibrated versions and non-overlapping progression
- **NEW: Byram Fire Intensity Calculation**: Two methods (radiative and traditional) with validation
- **NEW: Propagation Speed Analysis**: 90th percentile distance for robust speed estimation
- **NEW: Fire Front Length Estimation**: MTG-resolution corrected length calculation
- **NEW: Radiative Efficiency**: Cross-validation between intensity methods

## 🛠️ Installation

### Prerequisites

```bash
# Install Python dependencies (shapely 2.x)
pip install pandas geopandas shapely scikit-learn numpy
```

### Download Script

```bash
git clone https://github.com/PedroVenancio/mtg-frp-fire-progression.git
cd mtg-fire-progression
```

## 🚀 Usage

### Basic Examples

**Process fire progression with default parameters:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix frp_fires
```

**Process with calibration using reference areas:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix calibrated_frp_fires --reference_areas reference_burned_areas.gpkg
```

**Process without non-overlapping progression:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix no_overlap --no_non_overlapping
```

**Process only cumulative progression:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix cumulative_only --no_hourly
```

**Custom FRP filtering and clustering:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix custom --min_frp 15 --min_cluster_size 5 --density_eps 500
```

### Advanced Examples with Intensity Calculation

**Calculate Byram fire intensity with default parameters:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix fire_intensity --calculate_intensity
```

**Calculate intensity with custom radiative fraction and fuel type:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix custom_intensity \
    --calculate_intensity \
    --radiative_fraction 0.17 \
    --fuel_type forest \
    --mtg_correction 0.65
```

**Complete workflow with calibration and intensity:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix complete_analysis \
    --calculate_intensity \
    --reference_areas reference_burned_areas.gpkg \
    --radiative_fraction 0.15 \
    --fuel_type shrub \
    --min_area_non_overlapping 0.5
```

**Custom minimum area non-overlapping with intensity visualization:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix intensity_print \
    --calculate_intensity \
    --min_area_non_overlapping 1.0 \
    --radiative_fraction 0.16
```

### Advanced Examples

**Fine-tune hull generation and area correction:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix tuned \
    --ratio 0.05 \
    --buffer_distance 100 \
    --shrink_distance 40 \
    --frp_threshold_method adaptive \
    --frp_quantile_threshold 0.2
```

**Complete workflow with calibration and custom non-overlapping area:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix complete \
    --reference_areas reference_burned_areas.gpkg \
    --min_area_non_overlapping 0.5 \
    --buffer_distance 80 \
    --shrink_distance 30
```

**Skip calibration even with reference data:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix no_calib \
    --reference_areas reference.gpkg \
    --skip_calibration
```

**Process with start hour instead of end hour:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix start_hour --use_start_hour
```

**Disable missing hour filling:**
```bash
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix no_fill --no_fill_missing
```

## 📊 Parameters

### Required Parameters
- `--input`: Input FRP data file (GeoPackage). Expects a GeoPackage generated by [MTG/MTFRPPixel Data Processor](https://github.com/PedroVenancio/mtg-frp-processor)
- `--output_prefix`: Prefix for all output files

### Core Processing Parameters
- `--fire_id_column`: Column identifying each fire (default: 'fire_id')
- `--buffer_distance`: Buffer distance in meters (default: 80)
- `--min_frp`: Minimum FRP value for filtering (default: 10)
- `--ratio`: Concave hull ratio parameter 0-1 (default: 0.08)
- `--min_cluster_size`: Minimum cluster size for polygons (default: 3)
- `--density_eps`: DBSCAN epsilon distance in meters (default: 300)
- `--shrink_distance`: Negative buffer distance in meters (default: 30)

### Non-Overlapping Progression Parameters
- `--no_non_overlapping`: Disable non-overlapping progression (enabled by default)
- `--min_area_non_overlapping`: Minimum area in hectares for non-overlapping polygons (default: 0.5)

### Byram Intensity Parameters (NEW)
- `--calculate_intensity`: Calculate Byram fire intensity (propagation speed and FRP in new area)
- `--radiative_fraction`: Radiative fraction for Byram intensity calculation. Xr = 0.15-0.20 for wildfires (based on Wooster et al., 2005; Johnston et al., 2017) (default: 0.15)
- `--fuel_consumption`: Fixed fuel consumption in kg/m². If None, uses speed-adjusted values (default: None)
- `--fuel_type`: Fuel type for continuous consumption model: 'grass', 'shrub', or 'forest' (default: shrub)
- `--mtg_correction`: Correction factor for MTG 1km resolution artifacts. Typical values: 0.6-0.7. Lower values reduce L more aggressively (default: 0.6)

### Processing Flags
- `--no_cumulative`: Disable cumulative progression
- `--no_hourly`: Disable hourly progression  
- `--no_fill_missing`: Disable filling missing hours
- `--no_density_filter`: Disable density filtering
- `--use_start_hour`: Use start hour instead of end hour

### Calibration Parameters
- `--reference_areas`: Reference burned areas file for calibration
- `--skip_calibration`: Skip calibration even if reference areas provided
- `--frp_threshold_method`: FRP threshold method: 'fixed' or 'adaptive' (default: adaptive)
- `--frp_quantile_threshold`: Quantile for adaptive FRP threshold (default: 0.15)

## 🗂️ Output Structure

### Generated Files
- `{prefix}_cumulative_initial.gpkg` - Initial cumulative progression
- `{prefix}_hourly_initial.gpkg` - Initial hourly progression  
- `{prefix}_cumulative_calibrated.gpkg` - Calibrated cumulative progression (if calibration applied)
- `{prefix}_hourly_calibrated.gpkg` - Calibrated hourly progression (if calibration applied)
- `{prefix}_daily_cumulative_initial.gpkg` - Daily summary of cumulative progression
- `{prefix}_daily_hourly_initial.gpkg` - Daily summary of hourly progression
- `{prefix}_daily_cumulative_calibrated.gpkg` - Calibrated daily summary (if calibration applied)
- `{prefix}_daily_hourly_calibrated.gpkg` - Calibrated daily summary (if calibration applied)

### Non-Overlapping Progression Files
- `{prefix}_non_overlapping_hourly_initial.gpkg` - Hourly non-overlapping progression (initial)
- `{prefix}_non_overlapping_hourly_calibrated.gpkg` - Hourly non-overlapping progression (calibrated)
- `{prefix}_daily_non_overlapping_initial.gpkg` - Daily non-overlapping summary (initial)
- `{prefix}_daily_non_overlapping_calibrated.gpkg` - Daily non-overlapping summary (calibrated)

### Byram Intensity Files (NEW)
- `{prefix}_byram_improved_initial.gpkg` - Cumulative progression with fire behavior and Byram intensity columns (initial)
- `{prefix}_byram_improved_calibrated.gpkg` - Cumulative progression with fire behavior and Byram intensity columns (calibrated)
- `{prefix}_fire_metrics_improved_initial.csv` - Detailed fire metrics CSV (initial)
- `{prefix}_fire_metrics_improved_calibrated.csv` - Detailed fire metrics CSV (calibrated)
- `{prefix}_propagation_vectors_90th_percentile.gpkg` - Propagation points using 90th percentile distance
- `{prefix}_propagation_vectors_max.gpkg` - Propagation points using maximum distance
- `{prefix}_non_overlapping_hourly_byram_improved_initial.gpkg` - Non-overlapping progression with intensity (initial)
- `{prefix}_non_overlapping_hourly_byram_improved_calibrated.gpkg` - Non-overlapping progression with intensity (calibrated)

### Data Structure
Files contain:
- `fire_id`: Fire identifier
- `datetime_hour`: Hour in UTC
- `datetime_display`: Display hour (end hour by default)
- `geometry`: Progression polygon
- `area_ha`: Area in hectares
- `n_points_current_hour`: Points in current hour
- `n_points_cumulative`: Cumulative points (cumulative version)
- `frp_current_hour`: FRP in current hour
- `frp_cumulative`: Cumulative FRP (cumulative version)
- `hull_type`: Type of hull used ('concave', 'convex', etc.)
- `data_status`: Data origin ('original', 'filled')
- `shrink_applied`: Whether negative buffer was applied
- `shrink_distance_m`: Shrink distance in meters

### NEW: Byram Intensity Fields
- `propagation_distance_m`: 90th percentile propagation distance (meters) - robust metric
- `propagation_distance_max_m`: Maximum propagation distance (meters) - absolute maximum
- `propagation_distance_reduction_pct`: Reduction percentage (90th vs max) - data quality indicator
- `propagation_speed_ms`: Propagation speed (rate of spread) (m/s)
- `propagation_speed_kmh`: Propagation speed (rate of spread) (km/h)
- `fire_front_length_m`: Fire front length with MTG correction (meters)
- `frp_new_area_mw`: FRP in new burned area only (MW)
- `n_points_new_area`: Number of FRP points in new area
- `byram_radiative_intensity_kw_m`: Radiative Byram intensity (kW/m) - I = FRP/(L×Xr)
- `byram_traditional_intensity_kw_m`: Traditional Byram intensity (kW/m) - I = H×w×v
- `radiative_efficiency`: Ratio radiative/traditional intensity (validation metric)
- `implied_radiative_fraction`: Implied Xr that would equalize both methods
- `intensity_class`: Radiative intensity classification
- `traditional_intensity_class`: Traditional intensity classification
- `adjusted_fuel_consumption_kg_m2`: Speed-adjusted fuel consumption

### Non-Overlapping Specific Fields
- `area_ha_new`: New area burned in this time step (hectares)
- `area_ha_cumulative`: Total cumulative area up to this time step
- `progression_type`: 'non_overlapping' or 'daily_non_overlapping'

## 🔧 Processing Details

### Algorithm Overview

1. **Data Filtering**: 
   - FRP thresholding (fixed or adaptive)
   - Spatial density filtering (DBSCAN)
   - Cluster size filtering

2. **Temporal Processing**:
   - UTC timezone enforcement
   - Hourly binning
   - Missing hour filling (optional)

3. **Polygon Generation**:
   - Concave hull with progressive ratio
   - Convex hull fallback
   - Positive buffer for smoothing
   - Negative buffer for area correction

4. **Propagation Analysis (NEW)**:
   - Calculate 90th percentile propagation distance (robust)
   - Estimate fire front length with MTG correction
   - Calculate propagation speed (rate of spread) (m/s and km/h)
   - Generate propagation vectors for visualization

5. **Intensity Calculation (NEW)**:
   - **Radiative Method**: I_rad = FRP / (L × X_r)
   - **Traditional Method**: I_trad = H × w × v
   - Cross-validation through radiative efficiency
   - Intensity classification (Low to Extreme)

6. **Non-Overlapping Progression**:
   - Calculate geometric difference between consecutive time steps
   - Show only new burned areas for each hour
   - Apply minimum area filter (default: 0.5 ha)
   - Create daily summaries of new burned areas

7. **Calibration**:
   - Compare with reference burned areas
   - Calculate overestimation ratios
   - Suggest optimal shrink distance

### Byram Intensity Calculation Details

The intensity calculation system implements:

**Radiative Method (based on Wooster et al., 2005; Johnston et al., 2017):**
```
I_rad = FRP / (L × X_r)
```
Where:
- `I_rad` = Radiative Byram intensity (kW/m)
- `FRP` = Fire Radiative Power in new area (converted to kW)
- `L` = Fire front length (m) - MTG corrected
- `X_r` = Radiative fraction (default: 0.15, range: 0.15-0.20)

**Traditional Method (Byram, 1959):**
```
I_trad = H × w × v
```
Where:
- `I_trad` = Traditional Byram intensity (kW/m)
- `H` = Heat content (20,000 kJ/kg for Mediterranean fuels)
- `w` = Fuel consumption (kg/m²) - speed-adjusted or fixed
- `v` = Propagation speed (m/s)

**Key Innovations:**
- **90th percentile distance**: Uses 90th percentile instead of maximum for robust propagation
- **MTG correction**: Applies 0.6 factor to fire front length for 1km resolution artifacts
- **Speed-adjusted fuel consumption**: Fuel consumption varies with propagation speed
- **Validation metrics**: Radiative efficiency and implied X_r for cross-validation

### Calibration Process

The calibration system:
- Matches fires between FRP data and reference areas
- Calculates overestimation percentages
- Suggests optimal `shrink_distance` parameter
- Heuristic: 10 meters shrink per 10% overestimation
- Limits suggested shrink between 20-80 meters

### Non-Overlapping Progression (ENABLED BY DEFAULT)

The non-overlapping progression is now **enabled by default** because it provides cleaner visualization for most use cases:
- Shows only the **new area burned** in each time step
- Ideal for print layouts and progression visualization
- Filters out small polygons (< 0.5 ha by default)
- Maintains temporal sequence without overlaps
- Provides cleaner visualization of fire spread patterns

To disable this feature, use: `--no_non_overlapping`

## 💡 Usage Tips

### For Best Results

**Data Preparation:**
- Get FRP data from [MTG/MTFRPPixel Data Processor](https://github.com/PedroVenancio/mtg-frp-processor)
- Ensure consistent fire_id between FRP and reference data
- Use projected CRS (meters) for accurate distance calculations
- Clean invalid geometries before processing

**Parameter Tuning:**
- Start with `ratio=0.08` and adjust based on point density
- Use `adaptive` FRP threshold for datasets with varying fire intensities
- Set `min_cluster_size=2` for detecting small fires
- Increase `density_eps` for more dispersed fire patterns
- Use `min_area_non_overlapping=1.0` for cleaner print layouts

**Intensity Calculation:**
- Use `--calculate_intensity` for fire behavior analysis
- Adjust `--mtg_correction` based on landscape complexity (0.5-0.7)
- Monitor `propagation_distance_reduction_pct` for data quality
- Use `--fuel_type` appropriate for your vegetation (grass, shrub or forest)
- Validate with `radiative_efficiency` (should be 0.1-0.3)

**Memory Management:**
- Process large datasets in batches by fire_id
- Use `--no_hourly` or `--no_cumulative` to reduce output size
- Monitor RAM usage with very large input files

### QGIS Integration

1. **Visualization**:
   - Style by `datetime_hour` or `datetime_display` for temporal visualization and animation 
   - Use graduated symbols for `area_ha` or `byram_radiative_intensity_kw_m`
   - Filter by `data_status` to identify filled hours

2. **Animation**:
   - Use cumulative version with `datetime_hour` or `datetime_display` for temporal animation using QGIS Temporal Controller Panel
   - Daily summaries provide cleaner animation frames for large time frames

3. **Print Layouts**:
   - Use **non-overlapping progression** for clear progression maps
   - Each time step shows only new burned area
   - Perfect for static maps showing fire spread sequence
   - Color by intensity class for behavior analysis [Low (<500 kW/m); Moderate (500-2000 kW/m); 'High (2000-4000 kW/m)'; 'Very High (4000-10.000 kW/m)'; 'Extreme (>10.000 kW/m)']

4. **Analysis**:
   - Hourly version for detailed temporal analysis
   - Compare calibrated vs initial versions for accuracy assessment
   - Intersect with reference data to get better statistics and make it closer to the reality
   - Use intensity metrics for fire behavior studies

## 🐛 Troubleshooting

### Common Issues

**No polygons created:**
- Check FRP filtering thresholds
- Verify minimum cluster size
- Ensure valid fire_id values (not -1 or NaN)

**Overestimation issues:**
- Increase `shrink_distance`
- Use calibration with reference areas
- Adjust `ratio` parameter for tighter hulls

**Memory errors:**
- Process fires individually
- Increase `min_cluster_size` to reduce polygon count
- Use `--no_fill_missing` to reduce output size

**Temporal gaps:**
- Enable `fill_missing_hours` (default)
- Check input data temporal continuity
- Verify timezone handling

**Too many small polygons in non-overlapping output:**
- Increase `min_area_non_overlapping` (try 1.0 or 2.0 ha)
- Check if `shrink_distance` is appropriate
- Verify FRP filtering thresholds

**Intensity calculation issues:**
- Check `propagation_distance_reduction_pct` - values > 50% indicate data issues
- Adjust `mtg_correction` if fire front lengths seem unrealistic
- Verify if `radiative_efficiency` is in plausible range
- Check if fuel type matches the burned vegetation

### Performance Optimization

**For large datasets:**
```bash
# Process with stricter filtering
python mtg_frp_fire_progression.py --input large_data.gpkg --output_prefix optimized \
    --min_cluster_size 5 \
    --min_frp 20 \
    --no_hourly
```

**For high precision:**
```bash
# Use tighter parameters and calibration
python mtg_frp_fire_progression.py --input precise_data.gpkg --output_prefix precise \
    --ratio 0.05 \
    --buffer_distance 50 \
    --reference_areas high_quality_reference.gpkg
```

**For print-ready progression maps:**
```bash
# Generate clean non-overlapping progression
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix print_ready \
    --min_area_non_overlapping 1.0 \
    --buffer_distance 80 \
    --shrink_distance 40
```

**For detailed fire behavior analysis:**
```bash
# Generate complete intensity analysis
python mtg_frp_fire_progression.py --input fire_data.gpkg --output_prefix behavior \
    --calculate_intensity \
    --radiative_fraction 0.16 \
    --fuel_type forest \
    --mtg_correction 0.65 \
    --reference_areas reference_burned_areas.gpkg
```

## 📝 Complete Workflow Example

```bash
# 1. Process with calibration, non-overlapping, and intensity calculation
python mtg_frp_fire_progression.py --input mtg_frp_data.gpkg --output_prefix fire_analysis \
    --calculate_intensity \
    --reference_areas sentinel2_burned_areas.gpkg \
    --radiative_fraction 0.15 \
    --fuel_type shrub \
    --mtg_correction 0.6

# 2. Analyze results in QGIS
# Files: fire_analysis_*.gpkg and fire_analysis_*.csv

# 3. Compare intensity methods
# Open fire_analysis_non_overlapping_hourly_byram_improved_calibrated.gpkg
# Compare byram_radiative_intensity_kw_m vs byram_traditional_intensity_kw_m
# Check radiative_efficiency for validation

# 4. Visualize propagation vectors
# Open fire_analysis_propagation_vectors_90th_percentile.gpkg
# Style by propagation_speed_kmh for rate of spread visualization

# 5. Create temporal animation
# Use _cumulative_calibrated outputs with datetime_hour for animation

# 6. Create print layouts
# Use non_overlapping_hourly_byram_improved_calibrated for intensity progression maps
```

## 🔒 Data Requirements

### Input Data ([MTG/MTFRPPixel Data Processor](https://github.com/PedroVenancio/mtg-frp-processor))
- FRP data with geometry and fire_id column
- Valid datetime information (acquisition_datetime or ACQTIME)
- FRP values for filtering

### Reference Data (for calibration)
- Burned area polygons with matching fire_id
- Same spatial extent as FRP data
- Preferably from high-resolution sources (Sentinel-2, Landsat)

## 📄 License

This project is distributed under the MIT License. See LICENSE file for details.

## 🙋 Support

For issues and questions:
1. Check troubleshooting section
2. Create GitHub issue
3. Contact project maintainer

## 📚 References

### Core Algorithms
- [MTG/MTFRPPixel Data Processor](https://github.com/PedroVenancio/mtg-frp-processor)
- [LSA SAF Product Documentation](https://lsa-saf.eumetsat.int/en/news/news/2025/10/01/release-of-mtg-fire-radiative-power-product-as-demonstration/)
- [MTG Mission Overview](https://www.eumetsat.int/meteosat-third-generation)

### Intensity Calculation
- **Byram, G. M. (1959)**: Combustion of forest fuels. In: K. P. Davis (Editor), Forest fire control and use. McGraw-Hill, New York, pp. 61–89.
- **Wooster, M. J., Roberts, G., Freeborn, P. H., et al. (2005)**: Satellite remote sensing of fire radiative power for estimation of wildfire carbon flux and fuel consumption.
- **Johnston, J. M., Smith, A. M. S., & Wooster, M. J. (2017)**: Direct estimation of Byram’s fire intensity from infrared remote sensing imagery.
- **Fernandes, P., Loureiro, C. (2021)**: Modelos de combustível florestal para Portugal. Documento de referência, versão de 2021.

### GIS Integration
- [QGIS Documentation](https://qgis.org/resources/hub/)
- [QGIS Time-based control on the map canvas](https://docs.qgis.org/3.40/en/docs/user_manual/map_views/map_view.html#time-based-control-on-the-map-canvas)

---

**Notes**: This tool is designed to work with MTG FRP data but can be adapted for other fire detection datasets with similar structure. The intensity calculation methods are based on literature but have several assumptions that need to be validated for different fire regimes. As so, intensity values should be considered initial estimations to compare diferent wildfires.
