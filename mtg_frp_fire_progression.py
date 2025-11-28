import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from shapely import concave_hull
import numpy as np
from datetime import timedelta
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import argparse
import os
import sys

def create_non_overlapping_progression(progression_gdf, fire_id_column='fire_id', min_area_ha=0.5):
    """
    Create non-overlapping fire progression polygons where each hour shows only the NEW burned area.
    This is useful for print layouts and progression visualization.
    """
    print("=== CREATING NON-OVERLAPPING PROGRESSION ===")
    print(f"Minimum area filter: {min_area_ha} hectares")
    
    # Ensure data is sorted by fire_id and datetime
    progression_gdf = progression_gdf.sort_values([fire_id_column, 'datetime_hour'])
    
    non_overlapping_results = []
    
    for fire_id in progression_gdf[fire_id_column].unique():
        fire_data = progression_gdf[progression_gdf[fire_id_column] == fire_id].copy()
        fire_data = fire_data.sort_values('datetime_hour')
        
        print(f"Processing fire {fire_id} with {len(fire_data)} time steps...")
        
        previous_geometry = None
        polygons_created = 0
        polygons_filtered = 0
        
        for idx, row in fire_data.iterrows():
            current_geometry = row['geometry']
            current_hour = row['datetime_hour']
            current_display = row['datetime_display']
            
            # For the first hour, use the full geometry
            if previous_geometry is None:
                new_geometry = current_geometry
            else:
                # Calculate the difference: current - previous
                try:
                    if current_geometry.is_valid and previous_geometry.is_valid:
                        new_geometry = current_geometry.difference(previous_geometry)
                        
                        # Handle cases where difference might create empty or invalid geometries
                        if new_geometry.is_empty:
                            # Skip this hour if no new area
                            polygons_filtered += 1
                            previous_geometry = current_geometry
                            continue
                    else:
                        new_geometry = current_geometry
                except Exception as e:
                    print(f"  Error in geometry difference for fire {fire_id}, hour {current_hour}: {e}")
                    new_geometry = current_geometry
            
            # Calculate area of the new geometry
            if progression_gdf.crs and progression_gdf.crs.is_projected:
                new_area_ha = new_geometry.area / 10000
            else:
                new_area_ha = new_geometry.area * 111.32 * 111.32 * 100
            
            # Apply minimum area filter
            if new_area_ha < min_area_ha:
                polygons_filtered += 1
                previous_geometry = current_geometry
                continue
            
            # Create result record
            result = {
                'fire_id': fire_id,
                'datetime_hour': current_hour,
                'datetime_display': current_display,
                'geometry': new_geometry,
                'area_ha_new': new_area_ha,
                'area_ha_cumulative': row['area_ha'],
                'n_points_cumulative': row.get('n_points_cumulative', row.get('n_points', 0)),
                'frp_cumulative': row.get('frp_cumulative', row.get('frp', 0)),
                'hull_type': row.get('hull_type', 'unknown'),
                'data_status': row.get('data_status', 'original'),
                'progression_type': 'non_overlapping'
            }
            
            non_overlapping_results.append(result)
            previous_geometry = current_geometry
            polygons_created += 1
        
        print(f"  Fire {fire_id}: {polygons_created} non-overlapping polygons created, {polygons_filtered} filtered out (area < {min_area_ha} ha)")
    
    if not non_overlapping_results:
        print("No non-overlapping polygons were created.")
        return None
    
    non_overlapping_gdf = gpd.GeoDataFrame(non_overlapping_results, crs=progression_gdf.crs)
    
    # Additional filter for zero or negative area (shouldn't happen after min_area filter, but just in case)
    original_count = len(non_overlapping_gdf)
    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf['area_ha_new'] > 0]
    filtered_count = len(non_overlapping_gdf)
    
    if filtered_count < original_count:
        print(f"Additional filtering: {original_count - filtered_count} polygons with zero/negative area removed")
    
    print(f"Non-overlapping progression created: {len(non_overlapping_gdf)} polygons (min area: {min_area_ha} ha)")
    
    return non_overlapping_gdf

def create_daily_non_overlapping(non_overlapping_gdf, min_area_ha=0.5):
    """
    Create daily summary from non-overlapping hourly progression.
    Each day shows the union of all new areas from that day.
    """
    print("=== CREATING DAILY NON-OVERLAPPING SUMMARY ===")
    print(f"Minimum area filter: {min_area_ha} hectares")
    
    # Ensure UTC timezone
    for col in ['datetime_hour']:
        if col in non_overlapping_gdf.columns and non_overlapping_gdf[col].dt.tz is None:
            non_overlapping_gdf[col] = non_overlapping_gdf[col].dt.tz_localize('UTC')
    
    # Create date column
    non_overlapping_gdf['date'] = non_overlapping_gdf['datetime_hour'].dt.date
    
    daily_results = []
    days_created = 0
    days_filtered = 0
    
    for fire_id in non_overlapping_gdf['fire_id'].unique():
        fire_data = non_overlapping_gdf[non_overlapping_gdf['fire_id'] == fire_id]
        
        for date in fire_data['date'].unique():
            daily_data = fire_data[fire_data['date'] == date]
            
            if len(daily_data) == 0:
                continue
            
            # Union all new geometries from this day
            daily_geometries = daily_data['geometry'].tolist()
            
            try:
                daily_union = unary_union(daily_geometries)
                
                # Skip if union resulted in empty geometry
                if daily_union.is_empty:
                    days_filtered += 1
                    continue
                
                # Calculate daily area
                if non_overlapping_gdf.crs and non_overlapping_gdf.crs.is_projected:
                    daily_area_ha = daily_union.area / 10000
                else:
                    daily_area_ha = daily_union.area * 111.32 * 111.32 * 100
                
                # Apply minimum area filter
                if daily_area_ha < min_area_ha:
                    days_filtered += 1
                    continue
                
                # Get the last record of the day for other attributes
                last_record = daily_data.iloc[-1]
                
                # Create date fields
                date_start = pd.to_datetime(date).tz_localize('UTC')
                date_reference = date_start + timedelta(hours=23, minutes=59, seconds=59)
                
                daily_results.append({
                    'fire_id': fire_id,
                    'date': date_start,
                    'date_reference': date_reference,
                    'geometry': daily_union,
                    'daily_area_ha': daily_area_ha,
                    'cumulative_area_ha': last_record['area_ha_cumulative'],
                    'n_hours': len(daily_data),
                    'total_frp': last_record['frp_cumulative'],
                    'progression_type': 'daily_non_overlapping'
                })
                
                days_created += 1
                
            except Exception as e:
                print(f"Error creating daily union for fire {fire_id}, date {date}: {e}")
                days_filtered += 1
                continue
    
    if not daily_results:
        print("No daily non-overlapping polygons were created.")
        return None
    
    daily_gdf = gpd.GeoDataFrame(daily_results, crs=non_overlapping_gdf.crs)
    
    print(f"Daily non-overlapping summary created: {days_created} polygons, {days_filtered} filtered out (area < {min_area_ha} ha)")
    
    return daily_gdf

def create_hourly_fire_progression(gdf, fire_id_column='fire_id', buffer_distance=100, 
                                  cumulative=True, min_frp=None, ratio=0.1, use_end_hour=True,
                                  fill_missing_hours=True, frp_threshold_method='fixed',
                                  frp_quantile_threshold=0.15, min_cluster_size=3,
                                  use_density_filter=True, density_eps=500,
                                  shrink_distance=30):
    """
    Create hourly fire progression polygons using concave_hull as default
    with advanced filters to mitigate area overestimation.
    """
    
    # APPLY ADVANCED FRP FILTERS
    if min_frp is not None and 'FRP' in gdf.columns:
        original_count = len(gdf)
        
        if frp_threshold_method == 'adaptive':
            # Adaptive threshold based on data quantile
            frp_threshold = gdf['FRP'].quantile(frp_quantile_threshold)
            print(f"Adaptive FRP threshold: {frp_threshold:.2f} (quantile {frp_quantile_threshold})")
            gdf = gdf[gdf['FRP'] >= frp_threshold].copy()
        else:
            # Fixed threshold
            gdf = gdf[gdf['FRP'] >= min_frp].copy()
            frp_threshold = min_frp
        
        filtered_count = len(gdf)
        print(f"FRP filter: {filtered_count}/{original_count} points kept "
              f"({filtered_count/original_count*100:.1f}%)")
    
    # SPATIAL DENSITY FILTER (DBSCAN) - Remove isolated points
    if use_density_filter and len(gdf) > 0:
        original_count = len(gdf)
        
        # Convert coordinates to numpy array
        coords = np.array([[geom.x, geom.y] for geom in gdf.geometry])
        
        # Apply DBSCAN to remove isolated points
        db = DBSCAN(eps=density_eps, min_samples=2, metric='euclidean')
        labels = db.fit_predict(coords)
        
        # Keep only points that belong to clusters (not noise)
        cluster_mask = labels != -1
        gdf = gdf[cluster_mask].copy()
        
        filtered_count = len(gdf)
        print(f"Density filter: {filtered_count}/{original_count} points kept "
              f"({filtered_count/original_count*100:.1f}%)")
    
    # DATETIME AND TIMEZONE PREPARATION
    if 'acquisition_datetime' in gdf.columns:
        gdf['acquisition_datetime'] = pd.to_datetime(gdf['acquisition_datetime'])
    else:
        if 'ACQTIME' in gdf.columns:
            gdf['acquisition_datetime'] = pd.to_datetime(
                gdf['ACQTIME'], format='%Y%m%d%H%M%S', errors='coerce'
            )
        else:
            raise ValueError("No datetime column found.")
    
    # Remove rows with invalid datetime
    gdf = gdf.dropna(subset=['acquisition_datetime'])
    
    # FORCE UTC - remove any timezone information and treat as UTC
    if gdf['acquisition_datetime'].dt.tz is not None:
        gdf['acquisition_datetime'] = gdf['acquisition_datetime'].dt.tz_convert('UTC')
    else:
        # If no timezone, assume UTC
        gdf['acquisition_datetime'] = gdf['acquisition_datetime'].dt.tz_localize('UTC')
    
    # Create rounded date+hour column in UTC
    gdf['datetime_hour'] = gdf['acquisition_datetime'].dt.floor('h')
    
    # If use_end_hour=True, add 1 hour to represent end of interval
    if use_end_hour:
        gdf['datetime_display'] = gdf['datetime_hour'] + timedelta(hours=1)
    else:
        gdf['datetime_display'] = gdf['datetime_hour']
    
    results = []
    
    # CONFIGURATION AND CONVERSIONS
    print(f"Mode: {'CUMULATIVE' if cumulative else 'HOURLY'}")
    print(f"Using concave_hull with ratio={ratio} as default")
    print(f"Fill missing hours: {fill_missing_hours}")
    print(f"Density filter: {use_density_filter}")
    print(f"Negative buffer (shrink): {shrink_distance} meters")
    print(f"Dataset CRS: {gdf.crs}")
    print(f"Timezone: UTC (forced)")
    
    # Check if CRS is projected (meters) or geographic (degrees)
    if gdf.crs and gdf.crs.is_projected:
        print(f"Buffer distance: {buffer_distance} meters (projected CRS)")
        # For projected CRS, buffer is already in meters
        actual_buffer = buffer_distance
        actual_shrink = shrink_distance
    else:
        # For geographic CRS, convert meters to degrees (approximation)
        meters_to_degrees = buffer_distance / 111320  # 1 degree ≈ 111.32 km
        actual_buffer = meters_to_degrees
        actual_shrink = shrink_distance / 111320
        print(f"Buffer distance: {buffer_distance} meters ≈ {actual_buffer:.6f} degrees")
        print(f"Shrink distance: {shrink_distance} meters ≈ {actual_shrink:.6f} degrees")
    
    # PROCESSING BY FIRE
    for fire_id in gdf[fire_id_column].unique():
        if fire_id == -1 or pd.isna(fire_id):  # Discard noise clusters
            continue
            
        fire_data = gdf[gdf[fire_id_column] == fire_id].copy()
        
        # Sort by full date+hour
        fire_data = fire_data.sort_values('datetime_hour')
        
        print(f"Processing fire {fire_id} with {len(fire_data)} points...")
        
        # Determine complete temporal range if fill_missing_hours=True
        if fill_missing_hours and len(fire_data) > 0:
            start_hour = fire_data['datetime_hour'].min()
            end_hour = fire_data['datetime_hour'].max()
            # Create complete hourly range
            full_hour_range = pd.date_range(
                start=start_hour, 
                end=end_hour, 
                freq='h',
                tz='UTC'
            )
            print(f"  Complete temporal range: {start_hour} to {end_hour} ({len(full_hour_range)} hours)")
        else:
            full_hour_range = fire_data['datetime_hour'].unique()
        
        # CUMULATIVE VERSION: polygons accumulate progressively
        if cumulative:
            cumulative_points = []
            previous_polygon = None
            last_valid_result = None  # Store last valid result for filling
            
            # CRITICAL FIX: Ensure we have an initial polygon before starting filling
            initial_polygon_created = False
            
            # Iterate over complete hourly range (with filling) or only hours with data
            for hour in full_hour_range:
                hour_data = fire_data[fire_data['datetime_hour'] == hour]
                has_frp_data = len(hour_data) > 0
                
                # CASE 1: Hour without FRP data but with active filling
                if not has_frp_data and fill_missing_hours:
                    if last_valid_result is not None:
                        # Reuse last known polygon, updating only timestamp
                        display_hour = hour + timedelta(hours=1) if use_end_hour else hour
                        
                        results.append({
                            'fire_id': fire_id,
                            'datetime_hour': hour,
                            'datetime_display': display_hour,
                            'geometry': last_valid_result['geometry'],
                            'n_points_current_hour': 0,  # Zero points this hour
                            'n_points_cumulative': last_valid_result['n_points_cumulative'],  # Keep accumulated
                            'frp_current_hour': 0,  # Zero FRP this hour
                            'frp_cumulative': last_valid_result['frp_cumulative'],  # Keep accumulated
                            'area_ha': last_valid_result['area_ha'],  # Keep area
                            'hull_type': 'filled_missing',  # New type to identify filling
                            'ratio_value': None,
                            'data_status': 'filled',  # Fill indicator
                            'shrink_applied': last_valid_result.get('shrink_applied', False),
                            'shrink_distance_m': last_valid_result.get('shrink_distance_m', 0)
                        })
                    else:
                        # FIX: If no valid polygon and hour without data, don't create polygon
                        print(f"  Hour {hour}: no FRP data and no valid polygon - ignoring")
                    continue
                
                # CASE 2: Process hour WITH FRP data
                if has_frp_data:
                    # FIX: Check if we have enough points for this specific hour
                    # Don't use min_cluster_size here to avoid losing important initial hours
                    if len(hour_data) < 1:  # Just ensure at least 1 point
                        print(f"  Hour {hour}: no valid points - ignoring")
                        continue
                    
                    # Add this hour's points to cumulative
                    cumulative_points.extend(hour_data.geometry.tolist())
                    
                    # FIX: Use lower threshold for first polygon
                    min_points_for_hull = min_cluster_size
                    if not initial_polygon_created and len(cumulative_points) >= 2:
                        # Allow creating first polygon with fewer points
                        min_points_for_hull = 2
                    
                    if len(cumulative_points) >= min_points_for_hull:
                        try:
                            # Create MultiPoint
                            multi_point = MultiPoint(cumulative_points)
                            
                            # TRY MULTIPLE RATIOS PROGRESSIVELY
                            ratios_to_try = [ratio, ratio * 0.7, ratio * 0.5, ratio * 0.3]
                            current_hull = None
                            hull_type_used = 'none'
                            
                            for r in ratios_to_try:
                                try:
                                    test_hull = concave_hull(multi_point, ratio=r)
                                    if not test_hull.is_empty and test_hull.is_valid:
                                        current_hull = test_hull
                                        hull_type_used = 'concave'
                                        final_ratio = r
                                        break
                                except:
                                    continue
                            
                            # Fallback to convex hull if concave fails
                            if current_hull is None:
                                current_hull = multi_point.convex_hull
                                hull_type_used = 'convex'
                                final_ratio = None
                                print(f"  Fire {fire_id}, {hour}: Fallback to convex hull")
                            
                            # Apply positive buffers for smoothing
                            buffered_hull = current_hull.buffer(actual_buffer)
                            
                            # APPLY NEGATIVE BUFFER (SHRINK) to mitigate overestimation
                            if actual_shrink > 0:
                                try:
                                    shrunk_polygon = buffered_hull.buffer(-actual_shrink)
                                    if not shrunk_polygon.is_empty and shrunk_polygon.is_valid:
                                        final_polygon = shrunk_polygon
                                        shrink_applied = True
                                    else:
                                        final_polygon = buffered_hull
                                        shrink_applied = False
                                        print(f"  Shrink created invalid polygon - using original")
                                except Exception as e:
                                    final_polygon = buffered_hull
                                    shrink_applied = False
                                    print(f"  Shrink error: {e} - using original polygon")
                            else:
                                final_polygon = buffered_hull
                                shrink_applied = False
                            
                            # Combination with previous polygon (cumulative)
                            if previous_polygon is not None:
                                try:
                                    combined_polygon = unary_union([previous_polygon, final_polygon])
                                    current_polygon = combined_polygon
                                except Exception as e:
                                    print(f"  Polygon union error: {e} - using current polygon")
                                    current_polygon = final_polygon
                            else:
                                current_polygon = final_polygon
                            
                            previous_polygon = current_polygon
                            
                            # Metric calculation
                            hour_frp = hour_data['FRP'].sum() if 'FRP' in hour_data.columns else 0
                            total_frp = fire_data[fire_data['datetime_hour'] <= hour]['FRP'].sum() if 'FRP' in fire_data.columns else 0
                            
                            # Calculate area in HECTARES
                            if gdf.crs and gdf.crs.is_projected:
                                area_ha = current_polygon.area / 10000
                            else:
                                area_ha = current_polygon.area * 111.32 * 111.32 * 100
                            
                            display_hour = hour + timedelta(hours=1) if use_end_hour else hour
                            
                            current_result = {
                                'fire_id': fire_id,
                                'datetime_hour': hour,
                                'datetime_display': display_hour,
                                'geometry': current_polygon,
                                'n_points_current_hour': len(hour_data),
                                'n_points_cumulative': len(cumulative_points),
                                'frp_current_hour': hour_frp,
                                'frp_cumulative': total_frp,
                                'area_ha': area_ha,
                                'hull_type': hull_type_used,
                                'ratio_value': final_ratio if hull_type_used == 'concave' else None,
                                'data_status': 'original',
                                'shrink_applied': shrink_applied,
                                'shrink_distance_m': shrink_distance if shrink_applied else 0
                            }
                            
                            results.append(current_result)
                            last_valid_result = current_result
                            initial_polygon_created = True
                            
                        except Exception as e:
                            print(f"Error processing fire_id {fire_id}, hour {hour}: {e}")
                            # FIX: Try more robust fallback
                            try:
                                if len(cumulative_points) >= 2:
                                    multi_point = MultiPoint(cumulative_points)
                                    current_hull = multi_point.convex_hull
                                    current_polygon = current_hull.buffer(actual_buffer)
                                    
                                    if previous_polygon is not None:
                                        try:
                                            current_polygon = unary_union([previous_polygon, current_polygon])
                                        except:
                                            pass
                                    
                                    previous_polygon = current_polygon
                                    
                                    hour_frp = hour_data['FRP'].sum() if 'FRP' in hour_data.columns else 0
                                    total_frp = fire_data[fire_data['datetime_hour'] <= hour]['FRP'].sum() if 'FRP' in fire_data.columns else 0
                                    
                                    # Calculate area in HECTARES
                                    if gdf.crs and gdf.crs.is_projected:
                                        area_ha = current_polygon.area / 10000
                                    else:
                                        area_ha = current_polygon.area * 111.32 * 111.32 * 100
                                    
                                    display_hour = hour + timedelta(hours=1) if use_end_hour else hour
                                    
                                    current_result = {
                                        'fire_id': fire_id,
                                        'datetime_hour': hour,
                                        'datetime_display': display_hour,
                                        'geometry': current_polygon,
                                        'n_points_current_hour': len(hour_data),
                                        'n_points_cumulative': len(cumulative_points),
                                        'frp_current_hour': hour_frp,
                                        'frp_cumulative': total_frp,
                                        'area_ha': area_ha,
                                        'hull_type': 'convex_fallback_error',
                                        'ratio_value': None,
                                        'data_status': 'original',
                                        'shrink_applied': False,
                                        'shrink_distance_m': 0
                                    }
                                    
                                    results.append(current_result)
                                    last_valid_result = current_result
                                    initial_polygon_created = True
                                    
                            except Exception as e2:
                                print(f"  Critical fallback also failed: {e2}")
                    else:
                        # FIX: If not enough points but we have previous polygon,
                        # use last valid polygon to maintain continuity
                        if last_valid_result is not None:
                            display_hour = hour + timedelta(hours=1) if use_end_hour else hour
                            
                            results.append({
                                'fire_id': fire_id,
                                'datetime_hour': hour,
                                'datetime_display': display_hour,
                                'geometry': last_valid_result['geometry'],
                                'n_points_current_hour': len(hour_data),
                                'n_points_cumulative': len(cumulative_points),
                                'frp_current_hour': hour_data['FRP'].sum() if 'FRP' in hour_data.columns else 0,
                                'frp_cumulative': fire_data[fire_data['datetime_hour'] <= hour]['FRP'].sum() if 'FRP' in fire_data.columns else 0,
                                'area_ha': last_valid_result['area_ha'],
                                'hull_type': 'filled_insufficient_points',
                                'ratio_value': None,
                                'data_status': 'filled',
                                'shrink_applied': last_valid_result.get('shrink_applied', False),
                                'shrink_distance_m': last_valid_result.get('shrink_distance_m', 0)
                            })
                        else:
                            print(f"  Hour {hour}: only {len(cumulative_points)} accumulated points (minimum: {min_points_for_hull}) and no previous polygon - ignoring")
        
        # NON-CUMULATIVE VERSION: only current hour points
        else:
            last_polygon = None
            last_area = 0
            last_shrink_applied = False
            last_shrink_distance = 0
            
            for hour in full_hour_range:
                hour_data = fire_data[fire_data['datetime_hour'] == hour]
                has_frp_data = len(hour_data) > 0
                
                # CASE: Hour without FRP data but with active filling
                if not has_frp_data and fill_missing_hours and last_polygon is not None:
                    display_hour = hour + timedelta(hours=1) if use_end_hour else hour
                    
                    results.append({
                        'fire_id': fire_id,
                        'datetime_hour': hour,
                        'datetime_display': display_hour,
                        'geometry': last_polygon,
                        'n_points': 0,
                        'frp': 0,
                        'area_ha': last_area,
                        'hull_type': 'filled_missing',
                        'ratio_value': None,
                        'data_status': 'filled',
                        'shrink_applied': last_shrink_applied,
                        'shrink_distance_m': last_shrink_distance
                    })
                    continue
                
                # CASE: Process hour WITH FRP data
                if has_frp_data and len(hour_data) >= min_cluster_size:
                    try:
                        # Only this specific hour's points
                        multi_point = MultiPoint(hour_data.geometry.tolist())
                        
                        # Try concave_hull first
                        current_hull = concave_hull(multi_point, ratio=ratio)
                        hull_type_used = 'concave'
                        
                        if current_hull.is_empty or not current_hull.is_valid:
                            current_hull = multi_point.convex_hull
                            hull_type_used = 'convex'
                        
                        current_polygon = current_hull.buffer(actual_buffer)
                        
                        # Apply shrink if specified
                        if actual_shrink > 0:
                            try:
                                shrunk_polygon = current_polygon.buffer(-actual_shrink)
                                if not shrunk_polygon.is_empty and shrunk_polygon.is_valid:
                                    current_polygon = shrunk_polygon
                                    shrink_applied = True
                                else:
                                    shrink_applied = False
                            except:
                                shrink_applied = False
                        else:
                            shrink_applied = False
                        
                        last_polygon = current_polygon  # Store for possible filling
                        
                        hour_frp = hour_data['FRP'].sum() if 'FRP' in hour_data.columns else 0
                        
                        # Calculate area in HECTARES
                        if gdf.crs and gdf.crs.is_projected:
                            area_ha = current_polygon.area / 10000
                        else:
                            area_ha = current_polygon.area * 111.32 * 111.32 * 100
                        
                        last_area = area_ha  # Store for possible filling
                        last_shrink_applied = shrink_applied
                        last_shrink_distance = shrink_distance if shrink_applied else 0
                        
                        display_hour = hour + timedelta(hours=1) if use_end_hour else hour
                        
                        results.append({
                            'fire_id': fire_id,
                            'datetime_hour': hour,
                            'datetime_display': display_hour,
                            'geometry': current_polygon,
                            'n_points': len(hour_data),
                            'frp': hour_frp,
                            'area_ha': area_ha,
                            'hull_type': hull_type_used,
                            'ratio_value': ratio if hull_type_used == 'concave' else None,
                            'data_status': 'original',
                            'shrink_applied': shrink_applied,
                            'shrink_distance_m': shrink_distance if shrink_applied else 0
                        })
                        
                    except Exception as e:
                        print(f"Error processing fire_id {fire_id}, hour {hour}: {e}")
                        continue
        
        # PROCESSED FIRE STATISTICS
        fire_results = [r for r in results if r['fire_id'] == fire_id]
        polygons_count = len(fire_results)
        
        if polygons_count > 0:
            # Calculate detailed statistics
            hours_with_data = len([r for r in fire_results if r['data_status'] == 'original'])
            hours_filled = len([r for r in fire_results if r['data_status'] == 'filled'])
            hours_missing = len(full_hour_range) - polygons_count
            
            print(f"Fire {fire_id}: {polygons_count} polygons created "
                  f"(data: {hours_with_data}, filled: {hours_filled}, missing: {hours_missing})")
            
            if hours_missing > 0:
                print(f"  WARNING: {hours_missing} hours without polygons in temporal range")
        else:
            print(f"Fire {fire_id}: 0 polygons created")
    
    if not results:
        print("No polygons were created.")
        return None
    
    progression_gdf = gpd.GeoDataFrame(results, crs=gdf.crs)
    
    # Ensure datetime columns in result are also in UTC
    for col in ['datetime_hour', 'datetime_display']:
        if col in progression_gdf.columns:
            if progression_gdf[col].dt.tz is None:
                progression_gdf[col] = progression_gdf[col].dt.tz_localize('UTC')
    
    return progression_gdf


def create_daily_summary(progression_gdf):
    """
    Create daily summary of fire progression.
    Ensure only one polygon per day per fire (last of the day).
    """
    # Ensure UTC
    for col in ['datetime_hour']:
        if col in progression_gdf.columns and progression_gdf[col].dt.tz is None:
            progression_gdf[col] = progression_gdf[col].dt.tz_localize('UTC')
    
    # Create date column
    progression_gdf['date'] = progression_gdf['datetime_hour'].dt.date
    
    daily_data = []
    
    for fire_id in progression_gdf['fire_id'].unique():
        fire_data = progression_gdf[progression_gdf['fire_id'] == fire_id]
        
        # Group by date
        for date in fire_data['date'].unique():
            daily_fire_data = fire_data[fire_data['date'] == date]
            
            # Sort by datetime_hour to ensure we get the last of the day
            daily_fire_data = daily_fire_data.sort_values('datetime_hour')
            
            # Get ONLY the last polygon of the day (most recent)
            last_row = daily_fire_data.iloc[-1]
            
            # Use polygon from last record of the day
            last_polygon = last_row['geometry']
            max_area = last_row['area_ha']
            total_frp = last_row['frp_cumulative'] if 'frp_cumulative' in last_row else last_row['frp']
            
            # Create two date fields:
            date_start = pd.to_datetime(date).tz_localize('UTC')  # 00:00:00 UTC
            date_reference = date_start + timedelta(hours=23, minutes=59, seconds=59)  # 23:59:59 UTC
            
            daily_data.append({
                'fire_id': fire_id,
                'date': date_start,           # 00:00:00 UTC (for animation)
                'date_reference': date_reference,  # 23:59:59 UTC (for reference)
                'geometry': last_polygon,
                'max_area_ha': max_area,
                'total_frp': total_frp,
                'n_hours': len(daily_fire_data),
                'data_status': last_row.get('data_status', 'original'),
                'shrink_applied': last_row.get('shrink_applied', False),
                'shrink_distance_m': last_row.get('shrink_distance_m', 0)
            })
    
    daily_gdf = gpd.GeoDataFrame(daily_data, crs=progression_gdf.crs)
    
    # Verification: ensure no duplicates by fire_id and date
    duplicate_check = daily_gdf.duplicated(subset=['fire_id', 'date'], keep=False)
    if duplicate_check.any():
        print(f"WARNING: Found {duplicate_check.sum()} duplicate records by fire_id and date")
        daily_gdf = daily_gdf.drop_duplicates(subset=['fire_id', 'date'], keep='last')
    
    print(f"Daily summary created: {len(daily_gdf)} polygons (1 per day per fire)")
    
    # Daily summary statistics
    data_status_counts = daily_gdf['data_status'].value_counts()
    shrink_counts = daily_gdf['shrink_applied'].value_counts()
    print(f"Data status in summary: {data_status_counts.to_dict()}")
    print(f"Shrink applied in summary: {shrink_counts.to_dict()}")
    
    return daily_gdf


def calibrate_with_reference_areas(frp_gdf, reference_gdf, fire_id_column='fire_id'):
    """
    Calibrate parameters based on reference areas (Sentinel-2, Landsat, etc.)
    """
    print("=== CALIBRATION WITH REFERENCE AREAS ===")
    
    # Ensure same CRS
    if frp_gdf.crs != reference_gdf.crs:
        reference_gdf = reference_gdf.to_crs(frp_gdf.crs)
        print(f"Reference CRS converted to: {frp_gdf.crs}")
    
    calibration_results = []
    matched_fires = 0
    
    for fire_id in frp_gdf[fire_id_column].unique():
        frp_fire_data = frp_gdf[frp_gdf[fire_id_column] == fire_id]
        ref_fire_data = reference_gdf[reference_gdf[fire_id_column] == fire_id]
        
        if len(ref_fire_data) == 0:
            print(f"  Warning: No reference area for fire_id {fire_id}")
            continue
        
        matched_fires += 1
        
        # Find final polygon of fire (largest accumulated area)
        frp_final_polygon = frp_fire_data.iloc[-1]['geometry']
        ref_final_polygon = ref_fire_data.iloc[-1]['geometry']
        
        # Calculate areas in hectares
        if frp_gdf.crs.is_projected:
            frp_area = frp_final_polygon.area / 10000
            ref_area = ref_final_polygon.area / 10000
        else:
            frp_area = frp_final_polygon.area * 111.32 * 111.32 * 100
            ref_area = ref_final_polygon.area * 111.32 * 111.32 * 100
        
        # Calculate overestimation
        if ref_area > 0:
            overestimation_ratio = frp_area / ref_area
            overestimation_percent = (overestimation_ratio - 1) * 100
            calibration_factor = 1 / overestimation_ratio
        else:
            overestimation_ratio = 0
            overestimation_percent = 0
            calibration_factor = 1
        
        calibration_results.append({
            'fire_id': fire_id,
            'frp_area_ha': frp_area,
            'ref_area_ha': ref_area,
            'overestimation_ratio': overestimation_ratio,
            'overestimation_percent': overestimation_percent,
            'calibration_factor': calibration_factor
        })
        
        print(f"  Fire {fire_id}: FRP={frp_area:.1f} ha, Reference={ref_area:.1f} ha, "
              f"Overestimation={overestimation_percent:.1f}%")
    
    if calibration_results:
        cal_df = pd.DataFrame(calibration_results)
        avg_overestimation = cal_df['overestimation_percent'].mean()
        avg_calibration = cal_df['calibration_factor'].mean()
        
        print(f"\n=== CALIBRATION RESULTS ===")
        print(f"Calibrated fires: {matched_fires}/{len(frp_gdf[fire_id_column].unique())}")
        print(f"Average overestimation: {avg_overestimation:.1f}%")
        print(f"Average calibration factor: {avg_calibration:.3f}")
        
        # Suggest shrink_distance based on overestimation
        # Heuristic: 10 meters shrink for each 10% overestimation
        suggested_shrink = (avg_overestimation / 10) * 10
        suggested_shrink = max(20, min(80, suggested_shrink))  # Limit between 20-80 meters
        
        print(f"Suggested shrink_distance: {suggested_shrink:.0f} meters")
        
        return avg_calibration, suggested_shrink
    
    print("No reference data found for calibration.")
    return 1.0, 30  # Default values


def main():
    parser = argparse.ArgumentParser(description='Create fire progression polygons from FRP data with calibration')
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Input FRP data file (GeoPackage or Shapefile)')
    parser.add_argument('--output_prefix', required=True, help='Prefix for output files')
    
    # Optional arguments with defaults
    parser.add_argument('--fire_id_column', default='fire_id', help='Column identifying each fire (default: fire_id)')
    parser.add_argument('--buffer_distance', type=float, default=80, help='Buffer distance in meters (default: 80)')
    parser.add_argument('--min_frp', type=float, default=10, help='Minimum FRP value for filtering (default: 10)')
    parser.add_argument('--ratio', type=float, default=0.08, help='Concave hull ratio parameter (default: 0.08)')
    parser.add_argument('--min_cluster_size', type=int, default=3, help='Minimum cluster size for polygons (default: 3)')
    parser.add_argument('--density_eps', type=float, default=300, help='DBSCAN epsilon distance in meters (default: 300)')
    parser.add_argument('--shrink_distance', type=float, default=30, help='Negative buffer distance in meters (default: 30)')
    
    # NEW: Minimum area threshold for non-overlapping polygons
    parser.add_argument('--min_area_non_overlapping', type=float, default=0.5,
                       help='Minimum area in hectares for non-overlapping polygons (default: 0.5)')
    
    # Boolean flags
    parser.add_argument('--no_cumulative', action='store_true', help='Disable cumulative progression')
    parser.add_argument('--no_hourly', action='store_true', help='Disable hourly progression')
    parser.add_argument('--no_fill_missing', action='store_true', help='Disable filling missing hours')
    parser.add_argument('--no_density_filter', action='store_true', help='Disable density filtering')
    parser.add_argument('--use_start_hour', action='store_true', help='Use start hour instead of end hour')
    
    # NEW: Non-overlapping progression option
    parser.add_argument('--non_overlapping', action='store_true', 
                       help='Generate non-overlapping progression (useful for print layouts)')
    
    # Calibration
    parser.add_argument('--reference_areas', help='Reference areas file for calibration')
    parser.add_argument('--skip_calibration', action='store_true', help='Skip calibration even if reference areas provided')
    
    # Processing options
    parser.add_argument('--frp_threshold_method', choices=['fixed', 'adaptive'], default='adaptive', 
                       help='FRP threshold method (default: adaptive)')
    parser.add_argument('--frp_quantile_threshold', type=float, default=0.15,
                       help='Quantile for adaptive FRP threshold (default: 0.15)')
    
    args = parser.parse_args()
    
    print("=== FIRE PROGRESSION SYSTEM ===")
    
    # 1. LOAD FRP DATA
    print("\n1. Loading FRP data...")
    try:
        gdf = gpd.read_file(args.input)
        print(f"   Data loaded: {len(gdf)} points")
        print(f"   Fires identified: {gdf[args.fire_id_column].nunique()}")
    except Exception as e:
        print(f"   ERROR loading data: {e}")
        sys.exit(1)
    
    # Set processing flags based on arguments
    process_cumulative = not args.no_cumulative
    process_hourly = not args.no_hourly
    fill_missing = not args.no_fill_missing
    use_density_filter = not args.no_density_filter
    use_end_hour = not args.use_start_hour
    process_non_overlapping = args.non_overlapping
    
    results = {}
    
    # 2. INITIAL PROCESSING OF BOTH VERSIONS
    print("\n2. Initial processing of versions...")
    
    # 2.1 CUMULATIVE VERSION (for animation and non-overlapping)
    if process_cumulative:
        print("\n2.1 Generating initial CUMULATIVE progression...")
        progression_cumulative_initial = create_hourly_fire_progression(
            gdf, 
            fire_id_column=args.fire_id_column,
            buffer_distance=args.buffer_distance,
            cumulative=True,
            min_frp=args.min_frp,
            ratio=args.ratio,
            use_end_hour=use_end_hour,
            fill_missing_hours=fill_missing,
            frp_threshold_method=args.frp_threshold_method,
            frp_quantile_threshold=args.frp_quantile_threshold,
            min_cluster_size=args.min_cluster_size,
            use_density_filter=use_density_filter,
            density_eps=args.density_eps,
            shrink_distance=args.shrink_distance
        )
        
        if progression_cumulative_initial is not None:
            results['cumulative_initial'] = progression_cumulative_initial
            output_file = f"{args.output_prefix}_cumulative_initial.gpkg"
            progression_cumulative_initial.to_file(output_file, driver='GPKG')
            print(f"   Initial cumulative progression saved: {output_file}")
        else:
            print("   ERROR: No cumulative polygons were created.")
            process_cumulative = False
    
    # 2.2 HOURLY VERSION (for hourly analysis)
    if process_hourly:
        print("\n2.2 Generating initial HOURLY progression...")
        progression_hourly_initial = create_hourly_fire_progression(
            gdf,
            fire_id_column=args.fire_id_column,
            buffer_distance=args.buffer_distance,
            cumulative=False,
            min_frp=args.min_frp,
            ratio=args.ratio,
            use_end_hour=use_end_hour,
            fill_missing_hours=False,  # Only hours with FRP for hourly version
            frp_threshold_method=args.frp_threshold_method,
            frp_quantile_threshold=args.frp_quantile_threshold,
            min_cluster_size=args.min_cluster_size,
            use_density_filter=use_density_filter,
            density_eps=args.density_eps,
            shrink_distance=args.shrink_distance
        )
        
        if progression_hourly_initial is not None:
            results['hourly_initial'] = progression_hourly_initial
            output_file = f"{args.output_prefix}_hourly_initial.gpkg"
            progression_hourly_initial.to_file(output_file, driver='GPKG')
            print(f"   Initial hourly progression saved: {output_file}")
        else:
            print("   WARNING: No hourly polygons were created")
            process_hourly = False
    
    # 3. CALIBRATION WITH REFERENCE DATA (ONLY WITH CUMULATIVE VERSION)
    calibration_shrink = args.shrink_distance
    calibration_applied = False
    
    if args.reference_areas and not args.skip_calibration and process_cumulative:
        print("\n3. Running calibration with reference areas...")
        try:
            reference_gdf = gpd.read_file(args.reference_areas)
            print(f"   Reference data loaded: {len(reference_gdf)} polygons")
            
            # Run calibration on cumulative version
            calibration_factor, suggested_shrink = calibrate_with_reference_areas(
                results['cumulative_initial'], 
                reference_gdf,
                fire_id_column=args.fire_id_column
            )
            
            print(f"\n*** CALIBRATION RECOMMENDATION ***")
            print(f"Ideal shrink_distance: {suggested_shrink:.0f} meters")
            print(f"Area correction factor: {calibration_factor:.3f}")
            
            calibration_shrink = suggested_shrink
            
            # 4. REPROCESSING BOTH VERSIONS WITH CALIBRATED PARAMETERS (if needed)
            if suggested_shrink != args.shrink_distance:
                print(f"\n4. Reprocessing BOTH versions with shrink={suggested_shrink:.0f}m...")
                calibration_applied = True
                
                # 4.1 CALIBRATED CUMULATIVE VERSION
                if process_cumulative:
                    print("\n4.1 Generating calibrated CUMULATIVE progression...")
                    progression_cumulative_calibrated = create_hourly_fire_progression(
                        gdf, 
                        fire_id_column=args.fire_id_column,
                        buffer_distance=args.buffer_distance,
                        cumulative=True,
                        min_frp=args.min_frp,
                        ratio=args.ratio,
                        use_end_hour=use_end_hour,
                        fill_missing_hours=fill_missing,
                        frp_threshold_method=args.frp_threshold_method,
                        frp_quantile_threshold=args.frp_quantile_threshold,
                        min_cluster_size=args.min_cluster_size,
                        use_density_filter=use_density_filter,
                        density_eps=args.density_eps,
                        shrink_distance=suggested_shrink  # Calibrated shrink
                    )
                    
                    if progression_cumulative_calibrated is not None:
                        results['cumulative_calibrated'] = progression_cumulative_calibrated
                        output_file = f"{args.output_prefix}_cumulative_calibrated.gpkg"
                        progression_cumulative_calibrated.to_file(output_file, driver='GPKG')
                        print(f"   Calibrated cumulative progression saved: {output_file}")
                
                # 4.2 CALIBRATED HOURLY VERSION
                if process_hourly:
                    print("\n4.2 Generating calibrated HOURLY progression...")
                    progression_hourly_calibrated = create_hourly_fire_progression(
                        gdf,
                        fire_id_column=args.fire_id_column,
                        buffer_distance=args.buffer_distance,
                        cumulative=False,
                        min_frp=args.min_frp,
                        ratio=args.ratio,
                        use_end_hour=use_end_hour,
                        fill_missing_hours=False,
                        frp_threshold_method=args.frp_threshold_method,
                        frp_quantile_threshold=args.frp_quantile_threshold,
                        min_cluster_size=args.min_cluster_size,
                        use_density_filter=use_density_filter,
                        density_eps=args.density_eps,
                        shrink_distance=suggested_shrink  # Calibrated shrink
                    )
                    
                    if progression_hourly_calibrated is not None:
                        results['hourly_calibrated'] = progression_hourly_calibrated
                        output_file = f"{args.output_prefix}_hourly_calibrated.gpkg"
                        progression_hourly_calibrated.to_file(output_file, driver='GPKG')
                        print(f"   Calibrated hourly progression saved: {output_file}")
            else:
                print("\n4. Ideal shrink_distance equal to initial - keeping initial versions")
                
        except FileNotFoundError:
            print("\n*** WARNING: Reference file not found ***")
            print("   Calibration skipped. To use calibration:")
            print("   1. Create a reference areas file")
            print("   2. With polygons of actual burned areas") 
            print("   3. With matching 'fire_id' column")
            
        except Exception as e:
            print(f"\n*** ERROR in calibration: {e} ***")
            print("   Continuing with initial results...")
    
    # 5. CREATE NON-OVERLAPPING PROGRESSION (NEW FEATURE)
    if process_non_overlapping and process_cumulative:
        print("\n5. Creating non-overlapping progression...")
        
        # Use calibrated version if available, otherwise initial
        if calibration_applied and 'cumulative_calibrated' in results:
            cumulative_gdf = results['cumulative_calibrated']
            calibrated_suffix = "_calibrated"
        else:
            cumulative_gdf = results['cumulative_initial']
            calibrated_suffix = "_initial"
        
        # 5.1 Create hourly non-overlapping progression
        non_overlapping_hourly = create_non_overlapping_progression(
            cumulative_gdf,
            fire_id_column=args.fire_id_column,
            min_area_ha=args.min_area_non_overlapping
        )
        
        if non_overlapping_hourly is not None:
            results['non_overlapping_hourly'] = non_overlapping_hourly
            output_file = f"{args.output_prefix}_non_overlapping_hourly{calibrated_suffix}.gpkg"
            non_overlapping_hourly.to_file(output_file, driver='GPKG')
            print(f"   Non-overlapping hourly progression saved: {output_file}")
            
            # 5.2 Create daily non-overlapping summary
            daily_non_overlapping = create_daily_non_overlapping(
                non_overlapping_hourly,
                min_area_ha=args.min_area_non_overlapping
            )
            
            if daily_non_overlapping is not None:
                results['daily_non_overlapping'] = daily_non_overlapping
                output_file = f"{args.output_prefix}_daily_non_overlapping{calibrated_suffix}.gpkg"
                daily_non_overlapping.to_file(output_file, driver='GPKG')
                print(f"   Daily non-overlapping summary saved: {output_file}")
        else:
            print("   ERROR: No non-overlapping polygons were created")
    
    # 6. CREATE DAILY SUMMARIES FOR ALL VERSIONS
    print("\n6. Creating daily summaries...")
    
    if process_cumulative and 'cumulative_initial' in results:
        daily_summary_cumulative = create_daily_summary(results['cumulative_initial'])
        if calibration_applied and 'cumulative_calibrated' in results:
            output_file = f"{args.output_prefix}_daily_cumulative_calibrated.gpkg"
            daily_summary_cumulative.to_file(output_file, driver='GPKG')
            print(f"   Calibrated cumulative daily summary saved: {output_file}")
        else:
            output_file = f"{args.output_prefix}_daily_cumulative_initial.gpkg"
            daily_summary_cumulative.to_file(output_file, driver='GPKG')
            print(f"   Initial cumulative daily summary saved: {output_file}")
    
    if process_hourly and 'hourly_initial' in results:
        daily_summary_hourly = create_daily_summary(results['hourly_initial'])
        if calibration_applied and 'hourly_calibrated' in results:
            output_file = f"{args.output_prefix}_daily_hourly_calibrated.gpkg"
            daily_summary_hourly.to_file(output_file, driver='GPKG')
            print(f"   Calibrated hourly daily summary saved: {output_file}")
        else:
            output_file = f"{args.output_prefix}_daily_hourly_initial.gpkg"
            daily_summary_hourly.to_file(output_file, driver='GPKG')
            print(f"   Initial hourly daily summary saved: {output_file}")
    
    # 7. FINAL STATISTICS
    print("\n7. FINAL STATISTICS:")
    print(f"   Total fires processed: {gdf[args.fire_id_column].nunique()}")
    print(f"   Shrink_distance applied: {calibration_shrink} meters")
    print(f"   Calibration applied: {'YES' if calibration_applied else 'NO'}")
    print(f"   Non-overlapping progression: {'YES' if process_non_overlapping else 'NO'}")
    if process_non_overlapping:
        print(f"   Minimum area for non-overlapping: {args.min_area_non_overlapping} ha")
    
    if process_cumulative and 'cumulative_initial' in results:
        gdf_cumulative = results['cumulative_calibrated'] if calibration_applied else results['cumulative_initial']
        print(f"\n   CUMULATIVE VERSION:")
        print(f"   - Hourly polygons: {len(gdf_cumulative)}")
        
        if 'shrink_applied' in gdf_cumulative.columns:
            shrink_count = gdf_cumulative['shrink_applied'].sum()
            shrink_percent = (shrink_count / len(gdf_cumulative)) * 100
            print(f"   - Polygons with shrinkage: {shrink_count}/{len(gdf_cumulative)} ({shrink_percent:.1f}%)")
        
        if len(gdf_cumulative) > 0:
            avg_area = gdf_cumulative['area_ha'].mean()
            max_area = gdf_cumulative['area_ha'].max()
            print(f"   - Average area per polygon: {avg_area:.1f} ha")
            print(f"   - Maximum area: {max_area:.1f} ha")
    
    if process_hourly and 'hourly_initial' in results:
        gdf_hourly = results['hourly_calibrated'] if calibration_applied else results['hourly_initial']
        print(f"\n   HOURLY VERSION:")
        print(f"   - Hourly polygons: {len(gdf_hourly)}")
    
    if process_non_overlapping and 'non_overlapping_hourly' in results:
        gdf_non_overlapping = results['non_overlapping_hourly']
        print(f"\n   NON-OVERLAPPING VERSION:")
        print(f"   - Hourly polygons: {len(gdf_non_overlapping)}")
        
        if len(gdf_non_overlapping) > 0:
            avg_new_area = gdf_non_overlapping['area_ha_new'].mean()
            max_new_area = gdf_non_overlapping['area_ha_new'].max()
            print(f"   - Average new area per hour: {avg_new_area:.1f} ha")
            print(f"   - Maximum new area in one hour: {max_new_area:.1f} ha")
    
    print(f"\n=== PROCESSING COMPLETED SUCCESSFULLY ===")
    print(f"Generated files with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()