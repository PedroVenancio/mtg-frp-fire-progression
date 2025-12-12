import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from shapely import concave_hull
import numpy as np
import math
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
    print("\n=== CREATING NON-OVERLAPPING PROGRESSION ===")
    print(f"Minimum area filter: {min_area_ha} hectares")
    
    # Ensure data is sorted by fire_id and datetime
    progression_gdf = progression_gdf.sort_values([fire_id_column, 'datetime_hour'])
    
    non_overlapping_results = []
    
    for fire_id in progression_gdf[fire_id_column].unique():
        fire_data = progression_gdf[progression_gdf[fire_id_column] == fire_id].copy()
        fire_data = fire_data.sort_values('datetime_hour')
        
        print(f"\nProcessing fire {fire_id} with {len(fire_data)} time steps...")
        
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
    print("\n=== CREATING DAILY NON-OVERLAPPING SUMMARY ===")
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
        
        print(f"\nProcessing fire {fire_id} with {len(fire_data)} points...")
        
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


def create_daily_summary(progression_gdf, progression_type="cumulative"):
    """
    Create daily summary of fire progression.
    Ensure only one polygon per day per fire (last of the day).
    """
    print(f"\n=== CREATING DAILY SUMMARY ({progression_type.upper()}) ===")
    
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


def calculate_propagation_vector(previous_polygon, current_polygon):
    """
    Calculate the main propagation vector and statistics using the 90th percentile distance.
    Returns dictionary with multiple propagation metrics for robustness.
    """
    try:
        # 1. Calculate the difference to get only the new area burned this hour
        new_area = current_polygon.difference(previous_polygon)
        
        if new_area.is_empty:
            return {
                'distance_90th_percentile': 0.0,
                'distance_maximum': 0.0,
                'point_90th_percentile': None,
                'point_maximum': None,
                'new_area_geometry': new_area,
                'n_samples': 0,
                'all_distances': []
            }
        
        # 2. Sample points along the boundary and collect distances
        all_distances = []
        sample_points = []
        
        if hasattr(new_area, 'geoms'):
            # MultiPolygon - process each part
            for polygon in new_area.geoms:
                # Use only the exterior (outer ring) to avoid interior holes
                exterior = polygon.exterior
                num_points = max(20, int(exterior.length / 50))  # Sample every ~50m
                
                for i in range(num_points):
                    distance_along = (i / num_points) * exterior.length
                    point = exterior.interpolate(distance_along)
                    
                    # Calculate distance to the previous polygon
                    # This represents the fire front advancement
                    dist_to_previous = previous_polygon.distance(point)
                    
                    all_distances.append(dist_to_previous)
                    sample_points.append((point, dist_to_previous))
        else:
            # Single Polygon
            exterior = new_area.exterior
            num_points = max(20, int(exterior.length / 50))
            
            for i in range(num_points):
                distance_along = (i / num_points) * exterior.length
                point = exterior.interpolate(distance_along)
                
                dist_to_previous = previous_polygon.distance(point)
                
                all_distances.append(dist_to_previous)
                sample_points.append((point, dist_to_previous))
        
        if not all_distances:
            return {
                'distance_90th_percentile': 0.0,
                'distance_maximum': 0.0,
                'point_90th_percentile': None,
                'point_maximum': None,
                'new_area_geometry': new_area,
                'n_samples': 0,
                'all_distances': []
            }
        
        # 3. Calculate both 90th percentile and maximum distances
        import numpy as np
        distances_array = np.array(all_distances)
        
        distance_90th_percentile = np.percentile(distances_array, 90)
        distance_maximum = np.max(distances_array)
        
        # 4. Find corresponding points
        # For 90th percentile: point closest to the percentile value
        idx_90th = np.argmin(np.abs(distances_array - distance_90th_percentile))
        point_90th_percentile = sample_points[idx_90th][0]
        
        # For maximum: point with maximum distance
        idx_max = np.argmax(distances_array)
        point_maximum = sample_points[idx_max][0]
        
        # 5. Calculate additional statistics
        mean_distance = np.mean(distances_array)
        median_distance = np.median(distances_array)
        std_distance = np.std(distances_array)
        
        return {
            'distance_90th_percentile': distance_90th_percentile,
            'distance_maximum': distance_maximum,
            'point_90th_percentile': point_90th_percentile,
            'point_maximum': point_maximum,
            'new_area_geometry': new_area,
            'n_samples': len(all_distances),
            'all_distances': all_distances,
            'mean_distance': mean_distance,
            'median_distance': median_distance,
            'std_distance': std_distance,
            'reduction_percentage': (1 - distance_90th_percentile / distance_maximum) * 100 if distance_maximum > 0 else 0
        }
    
    except Exception as e:
        print(f"Error in propagation vector calculation: {e}")
        return {
            'distance_90th_percentile': 0.0,
            'distance_maximum': 0.0,
            'point_90th_percentile': None,
            'point_maximum': None,
            'new_area_geometry': None,
            'n_samples': 0,
            'all_distances': []
        }


def adjust_fuel_consumption_by_speed(speed_ms, fuel_type='shrub'):
    """
    Adjust fuel consumption based on fire spread rate.
    Faster fires have lower residence time, thus lower fuel consumption.
    Hybrid model with smooth transitions between empirically anchor points.
    Uses smoothing between these points for continuous behavior.
    """
    import math

    # Convert to m/h for easier reasoning
    speed_mh = speed_ms * 3600
    
    # Handle zero/near-zero speeds
    if speed_mh < 1.0:  # Less than 1 m/h
        return 0.3  # Very low consumption for no-spread hours

    # FUEL TYPE PARAMETERS (anchor points: speed_mps, consumption)
    fuel_anchors = {
        'grass': [
            (1.0, 0.4),      # Just above zero
            (60.0, 0.5),     # Very slow threshold
            (300.0, 0.55),   # Slow threshold
            (600.0, 0.6),    # Moderate threshold (peak consumption) - Assuming an average fuel load of 0.7 kg/m²
            (1200.0, 0.55),  # Mid-fast (decreasing)
            (1800.0, 0.5),   # Fast threshold
            (3000.0, 0.45),   # Very fast
            (5000.0, 0.4)    # Extreme
        ],
        'shrub': [
            (1.0, 0.6),      # Just above zero
            (60.0, 0.8),     # Very slow threshold
            (300.0, 1.0),    # Slow threshold
            (600.0, 1.2),    # Moderate threshold (peak consumption) - Assuming an average fuel load of 1.5 kg/m²
            (1200.0, 1.1),   # Mid-fast (decreasing)
            (1800.0, 1.0),   # Fast threshold
            (3000.0, 0.9),   # Very fast
            (5000.0, 0.8)    # Extreme
        ],
        'forest': [
            (1.0, 0.4),      # Just above zero
            (60.0, 0.7),     # Very slow threshold
            (300.0, 0.85),   # Slow threshold
            (600.0, 1.0),    # Moderate threshold (peak consumption) - Assuming an average fuel load of 1.3 kg/m²
            (1200.0, 0.95),  # Mid-fast (decreasing)
            (1800.0, 0.9),   # Fast threshold
            (3000.0, 0.8),   # Very fast
            (5000.0, 0.75)   # Extreme

        ]
    }
    anchors = fuel_anchors.get(fuel_type, fuel_anchors['shrub'])
    
    # Find the appropriate segment
    for i in range(len(anchors) - 1):
        if anchors[i][0] <= speed_mh < anchors[i+1][0]:
            x1, y1 = anchors[i]
            x2, y2 = anchors[i+1]
            
            # Avoid division by zero
            if x2 == x1:
                return y1
            
            # Position within segment (0 to 1)
            t = (speed_mh - x1) / (x2 - x1)
            
            # Smoothstep interpolation (cubic)
            smooth_t = t * t * (3 - 2 * t)
            
            return y1 + (y2 - y1) * smooth_t
    
    # Beyond last anchor
    return max(0.1, anchors[-1][1] * (anchors[-1][0] / speed_mh))


def calculate_improved_fire_front_length(new_area_polygon, previous_polygon, propagation_distance, mtg_correction=0.6):
    """
    Improved method for estimating active fire front length for MTG data.
    Considers MTG 1km resolution artifacts and provides realistic estimates.
    
    Parameters:
    -----------
    mtg_correction : float
        Correction factor for MTG 1km resolution (default: 0.6 - 40% reduction for MTG artifacts)
    """
    try:
        if new_area_polygon is None or new_area_polygon.is_empty:
            return 0.0
        
        # If no previous polygon (first hour), use conservative estimate
        if previous_polygon is None or previous_polygon.is_empty:
            # For MTG resolution, use area-based estimate for first hour
            area = new_area_polygon.area
            # Estimate: L ≈ √(4*area) for roughly circular shapes
            return (4 * area) ** 0.5 * mtg_correction  # MTG correction factor
        
        # METHOD 1: Use minimum rotated rectangle (better for MTG blocky shapes)
        try:
            mrr = new_area_polygon.minimum_rotated_rectangle
            coords = list(mrr.exterior.coords)
            
            # Calculate all side lengths
            lengths = []
            for i in range(4):
                x1, y1 = coords[i]
                x2, y2 = coords[(i+1) % 4]
                lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            
            # Sort lengths - for MTG, fire front is typically the shorter dimension
            sorted_lengths = sorted(lengths)
            
            # MTG CORRECTION: Use weighted average favoring shorter sides
            # Weight: 70% for shorter sides, 30% for longer sides
            weighted_length = (
                sorted_lengths[0] * 0.35 +  # Shortest side
                sorted_lengths[1] * 0.35 +  # Second shortest
                sorted_lengths[2] * 0.15 +  # Second longest
                sorted_lengths[3] * 0.15    # Longest side
            )
            
            # METHOD 2: Area/Propagation distance (geometric approach)
            area = new_area_polygon.area
            if propagation_distance > 0:
                length_from_area = area / propagation_distance
            else:
                # Fallback: approximate as width of advancing area
                length_from_area = weighted_length * 0.8
            
            # METHOD 3: Boundary analysis (only if previous polygon exists)
            try:
                # Get shared boundary
                shared = new_area_polygon.intersection(previous_polygon.boundary)
                if not shared.is_empty:
                    total_boundary = new_area_polygon.boundary.length
                    shared_length = shared.length
                    boundary_based = total_boundary - shared_length
                else:
                    boundary_based = new_area_polygon.boundary.length * 0.5
            except:
                boundary_based = weighted_length
            
            # COMBINE METHODS with MTG-specific weights:
            # For MTG 1km resolution, favor geometric methods over boundary analysis
            combined_length = (
                weighted_length * 0.4 +      # Geometric method (best for blocky shapes)
                length_from_area * 0.3 +     # Area-based method
                boundary_based * 0.3         # Boundary method (downweighted)
            )
            
            # MTG RESOLUTION CORRECTION
            # 1km pixels create artificially long boundaries
            # mtg_correction = 0.6  # 40% reduction for MTG artifacts
            
            # Ensure reasonable bounds (not too small, not too large)
            corrected_length = combined_length * mtg_correction
            
            # Lower bound: at least 500m (minimum realistic fire front for MTG detection)
            # Upper bound: not more than 5x the weighted geometric length
            min_length = 500  # 500 meters
            max_length = weighted_length * 5
            
            return max(min_length, min(corrected_length, max_length))
            
        except Exception as e:
            print(f"Error in improved fire front calculation: {e}")
            # Fallback: simple area-based estimate with MTG correction
            area = new_area_polygon.area
            return (area * 4) ** 0.5 * 0.6  # Conservative fallback
    
    except Exception as e:
        print(f"Error in fire front length calculation: {e}")
        return 1000  # Default 1 km fallback


def calculate_smart_propagation_speed(progression_gdf, frp_points_gdf, fire_id_column='fire_id', mtg_correction=0.6):
    """
    Calculate fire propagation speed AND fire front length.
    This method specifically tracks the fire front advancement.
    Uses 90th percentile distance for robust propagation estimation.
    """
    print("=== CALCULATING SMART PROPAGATION SPEED AND FIRE FRONT LENGTH ===")
    print("Using 90th percentile distance for robust propagation estimation")
    
    # Ensure data is sorted by fire_id and datetime
    progression_gdf = progression_gdf.sort_values([fire_id_column, 'datetime_hour'])
    
    # Ensure same CRS between progression polygons and FRP points
    if frp_points_gdf.crs != progression_gdf.crs:
        frp_points_gdf = frp_points_gdf.to_crs(progression_gdf.crs)
        print(f"FRP points CRS converted to: {progression_gdf.crs}")
    
    results = []
    
    for fire_id in progression_gdf[fire_id_column].unique():
        fire_data = progression_gdf[progression_gdf[fire_id_column] == fire_id].copy()
        fire_data = fire_data.sort_values('datetime_hour')
        
        print(f"Processing fire {fire_id} with {len(fire_data)} time steps...")
        
        previous_polygon = None
        previous_hour = None
        
        for idx, current_row in fire_data.iterrows():
            current_polygon = current_row['geometry']
            current_hour = current_row['datetime_hour']
            
            if previous_polygon is None:
                # First hour - no previous polygon, speed is zero
                results.append({
                    'fire_id': fire_id,
                    'datetime_hour': current_hour,
                    'propagation_distance_m': 0.0,
                    'propagation_distance_max_m': 0.0,
                    'propagation_speed_ms': 0.0,
                    'propagation_speed_kmh': 0.0,
                    'fire_front_length_m': 0.0,
                    'time_interval_h': 0.0,
                    'new_area_geometry': current_polygon,  # All area is new in first hour
                    'point_90th_percentile': None,
                    'point_maximum': None
                })
            else:
                try:
                    # Calculate propagation vector (fire front advancement) and statistics
                    propagation_stats = calculate_propagation_vector(previous_polygon, current_polygon)
                    
                    # Use 90th percentile distance for speed calculation
                    propagation_distance = propagation_stats['distance_90th_percentile']
                    max_distance = propagation_stats['distance_maximum']
                    
                    # Calculate fire front length
                    fire_front_length = calculate_improved_fire_front_length(
                        propagation_stats['new_area_geometry'], 
                        previous_polygon, 
                        propagation_distance,
                        mtg_correction=mtg_correction
                    )
                    
                    # Calculate time interval in hours
                    time_interval_h = (current_hour - previous_hour).total_seconds() / 3600.0
                    
                    if time_interval_h > 0 and propagation_distance > 0:
                        # Calculate speed in m/s and km/h
                        speed_ms = propagation_distance / (time_interval_h * 3600)
                        speed_kmh = propagation_distance / 1000 / time_interval_h
                    else:
                        speed_ms = 0.0
                        speed_kmh = 0.0
                    
                    results.append({
                        'fire_id': fire_id,
                        'datetime_hour': current_hour,
                        'propagation_distance_m': propagation_distance,
                        'propagation_distance_max_m': max_distance,
                        'propagation_speed_ms': speed_ms,
                        'propagation_speed_kmh': speed_kmh,
                        'propagation_distance_reduction_pct': propagation_stats.get('reduction_percentage', 0),
                        'fire_front_length_m': fire_front_length,
                        'time_interval_h': time_interval_h,
                        'new_area_geometry': propagation_stats['new_area_geometry'],
                        'point_90th_percentile': propagation_stats['point_90th_percentile'],
                        'point_maximum': propagation_stats['point_maximum'],
                        'propagation_n_samples': propagation_stats.get('n_samples', 0),
                        'propagation_mean_distance_m': propagation_stats.get('mean_distance', 0),
                        'propagation_median_distance_m': propagation_stats.get('median_distance', 0),
                        'propagation_std_distance_m': propagation_stats.get('std_distance', 0)
                    })
                    
                except Exception as e:
                    print(f"Error for fire {fire_id}, hour {current_hour}: {e}")
                    # Append zero values in case of error
                    results.append({
                        'fire_id': fire_id,
                        'datetime_hour': current_hour,
                        'propagation_distance_m': 0.0,
                        'propagation_distance_max_m': 0.0,
                        'propagation_speed_ms': 0.0,
                        'propagation_speed_kmh': 0.0,
                        'propagation_distance_reduction_pct': 0.0,
                        'fire_front_length_m': 0.0,
                        'time_interval_h': 0.0,
                        'new_area_geometry': None,
                        'point_90th_percentile': None,
                        'point_maximum': None,
                        'propagation_n_samples': 0,
                        'propagation_mean_distance_m': 0,
                        'propagation_median_distance_m': 0,
                        'propagation_std_distance_m': 0
                    })
            
            previous_polygon = current_polygon
            previous_hour = current_hour
    
    speed_df = pd.DataFrame(results)
    
    # Print statistics about propagation distances
    if len(speed_df) > 0:
        valid_distances = speed_df[speed_df['propagation_distance_m'] > 0]['propagation_distance_m']
        valid_max_distances = speed_df[speed_df['propagation_distance_max_m'] > 0]['propagation_distance_max_m']
        
        if len(valid_distances) > 0:
            print(f"\n   PROPAGATION DISTANCE STATISTICS (90th percentile):")
            print(f"   Mean propagation distance: {valid_distances.mean():.1f} m")
            print(f"   Median propagation distance: {valid_distances.median():.1f} m")
            print(f"   Max propagation distance: {valid_distances.max():.1f} m")
            
            if len(valid_max_distances) > 0:
                print(f"\n   COMPARISON WITH MAXIMUM DISTANCE:")
                print(f"   90th percentile mean: {valid_distances.mean():.1f} m")
                print(f"   Maximum distance mean: {valid_max_distances.mean():.1f} m")
                reduction = (1 - valid_distances.mean() / valid_max_distances.mean()) * 100
                print(f"   Reduction due to 90th percentile: {reduction:.1f}%")
                print(f"   This reduces influence of anomalous pixels")
    
    print(f"\nSmart propagation speed and fire front length calculation completed for {len(speed_df)} time steps")
    return speed_df


def calculate_frp_in_new_area(speed_df, frp_points_gdf, fire_id_column='fire_id'):
    """
    Calculate FRP only for points within the new burned area of each hour.
    This provides the correct FRP value for Byram intensity calculation.
    """
    print("\n=== CALCULATING FRP IN NEW BURNED AREA ===")
    
    # Check if 'datetime_hour' column exists in frp_points_gdf
    if 'datetime_hour' not in frp_points_gdf.columns:
        print("   'datetime_hour' column not found in FRP points. Creating it...")
        
        # Try to create datetime_hour from available datetime columns
        if 'acquisition_datetime' in frp_points_gdf.columns:
            frp_points_gdf['acquisition_datetime'] = pd.to_datetime(frp_points_gdf['acquisition_datetime'])
        elif 'ACQTIME' in frp_points_gdf.columns:
            frp_points_gdf['acquisition_datetime'] = pd.to_datetime(
                frp_points_gdf['ACQTIME'], format='%Y%m%d%H%M%S', errors='coerce'
            )
        else:
            raise ValueError("No datetime column found in FRP points.")
        
        # Remove rows with invalid datetime
        frp_points_gdf = frp_points_gdf.dropna(subset=['acquisition_datetime'])
        
        # Force UTC timezone
        if frp_points_gdf['acquisition_datetime'].dt.tz is not None:
            frp_points_gdf['acquisition_datetime'] = frp_points_gdf['acquisition_datetime'].dt.tz_convert('UTC')
        else:
            frp_points_gdf['acquisition_datetime'] = frp_points_gdf['acquisition_datetime'].dt.tz_localize('UTC')
        
        # Create rounded date+hour column in UTC
        frp_points_gdf['datetime_hour'] = frp_points_gdf['acquisition_datetime'].dt.floor('h')
        print(f"   Created 'datetime_hour' column for {len(frp_points_gdf)} points")
    
    results = []
    
    for idx, row in speed_df.iterrows():
        fire_id = row['fire_id']
        hour = row['datetime_hour']
        new_area_polygon = row['new_area_geometry']
        
        if new_area_polygon is None or new_area_polygon.is_empty:
            results.append({
                'fire_id': fire_id,
                'datetime_hour': hour,
                'frp_new_area_mw': 0.0,
                'n_points_new_area': 0
            })
            continue
        
        try:
            # Filter FRP points for this specific hour and fire
            hour_frp_points = frp_points_gdf[
                (frp_points_gdf[fire_id_column] == fire_id) & 
                (frp_points_gdf['datetime_hour'] == hour)
            ].copy()
            
            if len(hour_frp_points) == 0:
                results.append({
                    'fire_id': fire_id,
                    'datetime_hour': hour,
                    'frp_new_area_mw': 0.0,
                    'n_points_new_area': 0
                })
                continue
            
            # Check which points are within the new burned area
            frp_in_new_area = 0.0
            points_in_new_area = 0
            
            for point_idx, point_row in hour_frp_points.iterrows():
                if new_area_polygon.contains(point_row.geometry):
                    frp_in_new_area += point_row['FRP']
                    points_in_new_area += 1
            
            results.append({
                'fire_id': fire_id,
                'datetime_hour': hour,
                'frp_new_area_mw': frp_in_new_area,
                'n_points_new_area': points_in_new_area
            })
            
        except Exception as e:
            print(f"Error calculating FRP in new area for fire {fire_id}, hour {hour}: {e}")
            results.append({
                'fire_id': fire_id,
                'datetime_hour': hour,
                'frp_new_area_mw': 0.0,
                'n_points_new_area': 0
            })
    
    frp_new_area_df = pd.DataFrame(results)
    print(f"FRP in new area calculation completed for {len(frp_new_area_df)} time steps")
    return frp_new_area_df


def calculate_byram_with_radiative_fraction(progression_gdf, speed_df, frp_new_area_df,
                                           radiative_fraction=0.15,
                                           fuel_consumption=None,
                                           fuel_type='shrub',
                                           mtg_correction=0.6):
    """
    Calculate Byram fire intensity using the radiative fraction approach.
    I_B = FRP / (L * X_r)
    Where:
      I_B = Byram intensity (kW/m)
      FRP = Fire Radiative Power in the new burned area (kW)
      L = Fire front length (m) - already calculated in propagation speed analysis
      X_r = Radiative fraction (0.15-0.20 for wildfires)
      
    Also calculates traditional Byram intensity using fuel consumption and propagation speed.
    I_trad = H * w * v
    Where: 
      H = heat value (kJ/kg)
      w = fuel consumption (kg/m²)
      v = propagation speed (m/s)
    
    Parameters:
    -----------
    progression_gdf : GeoDataFrame
        Fire progression polygons with cumulative/hourly information
    speed_df : DataFrame
        Propagation speed and fire front length data from calculate_smart_propagation_speed()
    frp_new_area_df : DataFrame
        FRP values calculated only within new burned areas
    radiative_fraction : float, optional
        Radiative fraction X_r (default: 0.15, range: 0.15-0.20)
    fuel_consumption : float, optional
        Fixed fuel consumption in kg/m². If None, uses speed-adjusted values
    mtg_correction : float, optional
        Correction factor for MTG 1km resolution artifacts (default: 0.6)
    
    Returns:
    --------
    GeoDataFrame with Byram intensity columns added
    """
    print(f"\n=== CALCULATING BYRAM INTENSITY WITH RADIATIVE FRACTION {radiative_fraction} ===")
    
    # Merge all data sources
    # Include fire_front_length_m from speed_df (already calculated with MTG correction)
    merged_df = progression_gdf.merge(
        speed_df[[
            'fire_id', 'datetime_hour', 
            # Propagation metrics
            'propagation_speed_ms', 'propagation_speed_kmh', 
            'propagation_distance_m', 'propagation_distance_max_m',
            'propagation_distance_reduction_pct',
            # Fire front length
            'fire_front_length_m',
            # Geometry and points
            'new_area_geometry',
            'point_90th_percentile', 'point_maximum',
            # Statistics
            'propagation_n_samples', 'propagation_mean_distance_m',
            'propagation_median_distance_m', 'propagation_std_distance_m'
        ]],
        on=['fire_id', 'datetime_hour'],
        how='left'
    ).merge(
        frp_new_area_df[['fire_id', 'datetime_hour', 'frp_new_area_mw', 'n_points_new_area']],
        on=['fire_id', 'datetime_hour'],
        how='left'
    )
    
    # Fill missing values with 0 for numeric columns
    numeric_columns = [
        'propagation_speed_ms', 'propagation_speed_kmh', 
        'propagation_distance_m', 'propagation_distance_max_m',
        'propagation_distance_reduction_pct',
        'fire_front_length_m',
        'frp_new_area_mw', 'n_points_new_area',
        'propagation_n_samples', 'propagation_mean_distance_m',
        'propagation_median_distance_m', 'propagation_std_distance_m'
    ]
    
    # print(f"   Filling missing values for {len(numeric_columns)} numeric columns...")
    
    for col in numeric_columns:
        if col in merged_df.columns:
            # Count NaN values before fill
            # nan_count_before = merged_df[col].isna().sum()
            # if nan_count_before > 0:
                # print(f"      {col}: filling {nan_count_before} NaN values")
            merged_df[col] = merged_df[col].fillna(0)
    
    # Deal with geometric columns
    point_columns = ['point_90th_percentile', 'point_maximum', 'new_area_geometry']
    for col in point_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].where(merged_df[col].notna(), None)
    
    # CALCULATE RADIATIVE BYRAM INTENSITY: I_rad = FRP / (L * X_r)
    # Convert FRP from MW to kW (1 MW = 1000 kW)
    merged_df['byram_radiative_intensity_kw_m'] = 0.0
    # Calculate only for valid cases (L > 0 and X_r > 0)
    valid_mask = (merged_df['fire_front_length_m'] > 0) & (radiative_fraction > 0)
    
    if valid_mask.any():
        merged_df.loc[valid_mask, 'byram_radiative_intensity_kw_m'] = (
            merged_df.loc[valid_mask, 'frp_new_area_mw'] * 1000 / 
            (merged_df.loc[valid_mask, 'fire_front_length_m'] * radiative_fraction)
        )
    
    # Handle extreme values (replace inf/nan with 0)
    merged_df['byram_radiative_intensity_kw_m'] = (
        merged_df['byram_radiative_intensity_kw_m']
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    
    # CALCULATE TRADITIONAL BYRAM INTENSITY: I_trad = H * w * v
    # Where: H = heat value (kJ/kg), w = fuel consumption (kg/m²), v = propagation speed (m/s)
    
    # Use speed-adjusted fuel consumption if not provided
    if fuel_consumption is None:
        merged_df['adjusted_fuel_consumption_kg_m2'] = merged_df['propagation_speed_ms'].apply(
            lambda speed: adjust_fuel_consumption_by_speed(speed, fuel_type=fuel_type)
        )
        fuel_column = 'adjusted_fuel_consumption_kg_m2'
    else:
        merged_df['adjusted_fuel_consumption_kg_m2'] = fuel_consumption
        fuel_column = 'adjusted_fuel_consumption_kg_m2'
    
    heat_value = 20000  # kJ/kg - Average from portuguese fuel models (Fernandes et al.)
    
    merged_df['byram_traditional_intensity_kw_m'] = (
        heat_value * merged_df[fuel_column] * merged_df['propagation_speed_ms']
    )
    
    # CALCULATE RADIATIVE EFFICIENCY: ε = I_rad / I_trad
    merged_df['radiative_efficiency'] = 0.0
    # Calculate only when traditional intensity > 0
    efficiency_mask = (merged_df['byram_traditional_intensity_kw_m'] > 0)
    
    if efficiency_mask.any():
        merged_df.loc[efficiency_mask, 'radiative_efficiency'] = (
            merged_df.loc[efficiency_mask, 'byram_radiative_intensity_kw_m'] / 
            merged_df.loc[efficiency_mask, 'byram_traditional_intensity_kw_m']
        )
    
    # Limit radiative efficiency to realistic range (0-1)
    merged_df['radiative_efficiency'] = merged_df['radiative_efficiency'].clip(0, 1)
    
    # CALCULATE IMPLIED RADIATIVE FRACTION (for validation)
    merged_df['implied_radiative_fraction'] = 0.0
    
    # X_r_implied = FRP / (I_trad * L) when I_trad * L > 0
    implied_mask = (
        (merged_df['byram_traditional_intensity_kw_m'] > 0) &
        (merged_df['fire_front_length_m'] > 0)
    )
    
    if implied_mask.any():
        merged_df.loc[implied_mask, 'implied_radiative_fraction'] = (
            merged_df.loc[implied_mask, 'frp_new_area_mw'] * 1000 /
            (merged_df.loc[implied_mask, 'byram_traditional_intensity_kw_m'] * 
             merged_df.loc[implied_mask, 'fire_front_length_m'])
        )
    
    # Limit implied radiative fraction to reasonable range
    merged_df['implied_radiative_fraction'] = merged_df['implied_radiative_fraction'].clip(0, 0.5)
    
    # VALIDATION LOGGING: Check implied radiative fraction consistency
    if 'implied_radiative_fraction' in merged_df.columns:
        # Calculate statistics only for non-zero values
        non_zero_xr = merged_df[merged_df['implied_radiative_fraction'] > 0]['implied_radiative_fraction']
        
        if len(non_zero_xr) > 0:
            mean_xr = non_zero_xr.mean()
            median_xr = non_zero_xr.median()
            std_xr = non_zero_xr.std()
            
            print(f"\n IMPLIED RADIATIVE FRACTION VALIDATION:")
            print(f"   Records with non-zero Xr_implied: {len(non_zero_xr)}")
            print(f"   Mean Xr_implied: {mean_xr:.3f}")
            print(f"   Median Xr_implied: {median_xr:.3f}")
            print(f"   Std Dev: {std_xr:.3f}")
            print(f"   Expected Xr (input): {radiative_fraction:.3f}")
            
            # Threshold-based warnings
            if mean_xr > 0.25:
                print("\n   WARNING: Xr_implied consistently HIGH (> 0.25)\n")
                print("   Possible issues:")
                print("   1. Fire front length (L) may be UNDERestimated")
                print("   2. Fuel consumption (w) or speed (v) may be UNDERestimated")
                print("   3. FRP values may be OVERestimated")
                print("   4. Traditional intensity calculation may have issues")
            
            elif mean_xr < 0.10:
                print("\n   WARNING: Xr_implied consistently LOW (< 0.10)\n")
                print("   Possible issues:")
                print("   1. Fire front length (L) may be OVERestimated")
                print("   2. Fuel consumption (w) or speed (v) may be OVERestimated")
                print("   3. FRP values may be UNDERestimated")
                print("   4. MTG correction factor may be too aggressive")
            
            elif abs(mean_xr - radiative_fraction) > 0.05:
                print(f"\n   NOTE: Xr_implied differs from input by > 0.05")
                print(f"   Difference: {abs(mean_xr - radiative_fraction):.3f}")
                print("   Consider adjusting parameters or investigating data quality")
            
            else:
                print("\n    GOOD: Xr_implied within acceptable range of input value")
            
            # Additional diagnostic: distribution percentiles
            if len(non_zero_xr) >= 10:
                percentiles = non_zero_xr.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                print(f"\n   Distribution percentiles:")
                print(f"   10th: {percentiles[0.1]:.3f}, 25th: {percentiles[0.25]:.3f}, "
                      f"50th: {percentiles[0.5]:.3f}, 75th: {percentiles[0.75]:.3f}, "
                      f"90th: {percentiles[0.9]:.3f}")
    
    # INTENSITY CLASSIFICATION
    def classify_intensity(intensity):
        if intensity < 500:
            return 'Low (<500 kW/m)'
        elif intensity < 2000:
            return 'Moderate (500-2000 kW/m)'
        elif intensity < 4000:
            return 'High (2000-4000 kW/m)'
        elif intensity < 10000:
            return 'Very High (4000-10.000 kW/m)'
        else:
            return 'Extreme (>10.000 kW/m)'
    
    # Apply classification to radiative intensity
    merged_df['intensity_class'] = merged_df['byram_radiative_intensity_kw_m'].apply(classify_intensity)
    
    # Apply classification to traditional intensity
    merged_df['traditional_intensity_class'] = merged_df['byram_traditional_intensity_kw_m'].apply(classify_intensity)
    
    # Print comprehensive summary statistics for BOTH intensity methods
    print(f"\n   FINAL INTENSITY CALCULATION SUMMARY:")
    print(f"   Radiative fraction used: {radiative_fraction}")
    print(f"   Fuel consumption method: {'Speed-adjusted' if fuel_consumption is None else 'Fixed'}")
    print(f"   MTG correction factor: {mtg_correction}")
    
    # RADIATIVE INTENSITY STATISTICS
    radiative_mask = (
        (merged_df['byram_radiative_intensity_kw_m'] > 0) & 
        (merged_df['byram_radiative_intensity_kw_m'] < 100000) # Remove extreme outliers
    )
    radiative_intensity = merged_df[radiative_mask]['byram_radiative_intensity_kw_m']
    
    if len(radiative_intensity) > 0:
        print(f"\n   RADIATIVE BYRAM INTENSITY (I_rad = FRP / (L × X_r)):")
        print(f"   Valid records: {len(radiative_intensity)}")
        print(f"   Mean: {radiative_intensity.mean():.0f} kW/m")
        print(f"   Median: {radiative_intensity.median():.0f} kW/m")
        print(f"   Std Dev: {radiative_intensity.std():.0f} kW/m")
        print(f"   Min: {radiative_intensity.min():.0f} kW/m")
        print(f"   Max: {radiative_intensity.max():.0f} kW/m")
        
        # Radiative intensity class distribution - use the same mask
        radiative_class_counts = merged_df[radiative_mask]['intensity_class'].value_counts()
        
        print(f"\n   RADIATIVE INTENSITY CLASS DISTRIBUTION:")
        total_radiative = len(radiative_intensity)
        
        # Ensure all classes are represented, even if count is 0
        for class_name in ['Low (<500 kW/m)', 'Moderate (500-2000 kW/m)', 'High (2000-4000 kW/m)', 'Very High (4000-10.000 kW/m)', 'Extreme (>10.000 kW/m)']:
            count = radiative_class_counts.get(class_name, 0)
            percentage = (count / total_radiative) * 100 if total_radiative > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    # TRADITIONAL INTENSITY STATISTICS
    traditional_mask = (
        (merged_df['byram_traditional_intensity_kw_m'] > 0) & 
        (merged_df['byram_traditional_intensity_kw_m'] < 100000)
    )
    traditional_intensity = merged_df[traditional_mask]['byram_traditional_intensity_kw_m']
    
    if len(traditional_intensity) > 0:
        print(f"\n   TRADITIONAL BYRAM INTENSITY (I_trad = H × w × v):")
        print(f"   Valid records: {len(traditional_intensity)}")
        print(f"   Mean: {traditional_intensity.mean():.0f} kW/m")
        print(f"   Median: {traditional_intensity.median():.0f} kW/m")
        print(f"   Std Dev: {traditional_intensity.std():.0f} kW/m")
        print(f"   Min: {traditional_intensity.min():.0f} kW/m")
        print(f"   Max: {traditional_intensity.max():.0f} kW/m")
        
        # Traditional intensity class distribution - use the same mask
        traditional_class_counts = merged_df[traditional_mask]['traditional_intensity_class'].value_counts()
        
        print(f"\n   TRADITIONAL INTENSITY CLASS DISTRIBUTION:")
        total_traditional = len(traditional_intensity)
        
        for class_name in ['Low (<500 kW/m)', 'Moderate (500-2000 kW/m)', 'High (2000-4000 kW/m)', 'Very High (4000-10.000 kW/m)', 'Extreme (>10.000 kW/m)']:
            count = traditional_class_counts.get(class_name, 0)
            percentage = (count / total_traditional) * 100 if total_traditional > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    # COMPARISON BETWEEN METHODS
    if len(radiative_intensity) > 0 and len(traditional_intensity) > 0:
        print(f"\n   COMPARISON BETWEEN METHODS:")
        
        # Mean comparison
        mean_ratio = radiative_intensity.mean() / traditional_intensity.mean() if traditional_intensity.mean() > 0 else 0
        print(f"   Mean ratio (Radiative/Traditional): {mean_ratio:.3f}")
        
        # Median comparison
        median_ratio = radiative_intensity.median() / traditional_intensity.median() if traditional_intensity.median() > 0 else 0
        print(f"   Median ratio (Radiative/Traditional): {median_ratio:.3f}")
        
        # Correlation (when both > 0)
        valid_comparison = merged_df[
            (merged_df['byram_radiative_intensity_kw_m'] > 0) & 
            (merged_df['byram_traditional_intensity_kw_m'] > 0) &
            (merged_df['byram_radiative_intensity_kw_m'] < 50000) &
            (merged_df['byram_traditional_intensity_kw_m'] < 50000)
        ]
        
        if len(valid_comparison) > 1:
            correlation = valid_comparison['byram_radiative_intensity_kw_m'].corr(
                valid_comparison['byram_traditional_intensity_kw_m']
            )
            print(f"   Pearson correlation: {correlation:.3f}")
            
            # Classification agreement
            agreement_mask = valid_comparison['intensity_class'] == valid_comparison['traditional_intensity_class']
            agreement_percent = (agreement_mask.sum() / len(valid_comparison)) * 100
            print(f"   Classification agreement: {agreement_percent:.1f}%")
            
            # Detailed agreement matrix
            print(f"\n   CLASSIFICATION AGREEMENT MATRIX:")
            agreement_matrix = pd.crosstab(
                valid_comparison['intensity_class'], 
                valid_comparison['traditional_intensity_class'],
                rownames=['Radiative'], 
                colnames=['Traditional']
            )
            
            # Reorder columns and rows for consistency
            order = ['Low (<500 kW/m)', 'Moderate (500-2000 kW/m)', 'High (2000-4000 kW/m)', 'Very High (4000-10.000 kW/m)', 'Extreme (>10.000 kW/m)']
            agreement_matrix = agreement_matrix.reindex(index=order, columns=order, fill_value=0)
            
            for rad_class in order:
                if rad_class in agreement_matrix.index:
                    row = agreement_matrix.loc[rad_class]
                    row_str = "   "
                    for trad_class in order:
                        count = row[trad_class] if trad_class in row else 0
                        row_str += f"{trad_class}: {count:3d}  "
                    print(row_str)
    
    # RADIATIVE EFFICIENCY STATISTICS
    valid_efficiency = merged_df[
        (merged_df['radiative_efficiency'] > 0) & 
        (merged_df['radiative_efficiency'] <= 1)
    ]['radiative_efficiency']
    
    if len(valid_efficiency) > 0:
        print(f"\n   RADIATIVE EFFICIENCY (ε = I_rad / I_trad):")
        print(f"   Valid records: {len(valid_efficiency)}")
        print(f"   Mean: {valid_efficiency.mean():.3f}")
        print(f"   Median: {valid_efficiency.median():.3f}")
        print(f"   Std Dev: {valid_efficiency.std():.3f}")
        print(f"   Min: {valid_efficiency.min():.3f}")
        print(f"   Max: {valid_efficiency.max():.3f}")
        
        # Efficiency distribution
        efficiency_bins = {
            'Very Low (ε < 0.05)': (0, 0.05),
            'Low (0.05 ≤ ε < 0.10)': (0.05, 0.10),
            'Moderate (0.10 ≤ ε < 0.15)': (0.10, 0.15),
            'High (0.15 ≤ ε < 0.20)': (0.15, 0.20),
            'Very High (ε ≥ 0.20)': (0.20, 1.0)
        }
        
        print(f"\n   RADIATIVE EFFICIENCY DISTRIBUTION:")
        for label, (low, high) in efficiency_bins.items():
            if high == 1.0:
                count = len(valid_efficiency[(valid_efficiency >= low) & (valid_efficiency <= high)])
            else:
                count = len(valid_efficiency[(valid_efficiency >= low) & (valid_efficiency < high)])
            
            if len(valid_efficiency) > 0:
                percentage = (count / len(valid_efficiency)) * 100
                print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # FUEL CONSUMPTION STATISTICS (if speed-adjusted)
    if fuel_consumption is None:
        valid_fuel = merged_df[merged_df['adjusted_fuel_consumption_kg_m2'] > 0]['adjusted_fuel_consumption_kg_m2']
        if len(valid_fuel) > 0:
            # NON-ZERO SPEEDS ONLY
            print(f"\n   SPEED-ADJUSTED FUEL CONSUMPTION:")
            print(f"   Mean: {valid_fuel.mean():.2f} kg/m²")
            print(f"   Range: {valid_fuel.min():.2f} - {valid_fuel.max():.2f} kg/m²")

            # Calculate distribution for NON-ZERO speeds only
            non_zero_speeds = merged_df[merged_df['propagation_speed_ms'] > 0]
            total_non_zero = len(non_zero_speeds)

            if total_non_zero > 0:
                speed_mh = non_zero_speeds['propagation_speed_ms'] * 3600
                
                print(f"\n   DISTRIBUTION BY FIRE SPREAD RATE (NON-ZERO speeds only):")
                print(f"   Total non-zero speed records: {total_non_zero}")
                
                speed_categories = {
                    'Very slow (<60 m/hr)': len(speed_mh[speed_mh < 60]),
                    'Slow (60-300 m/hr)': len(speed_mh[(speed_mh >= 60) & (speed_mh < 300)]),
                    'Moderate (300-600 m/hr)': len(speed_mh[(speed_mh >= 300) & (speed_mh < 600)]),
                    'Fast (600-1800 m/hr)': len(speed_mh[(speed_mh >= 600) & (speed_mh < 1800)]),
                    'Very fast (1800-3600 m/hr)': len(speed_mh[(speed_mh >= 1800) & (speed_mh < 3600)]),
                    'Extreme (>3600 m/hr)': len(speed_mh[speed_mh >= 3600])
                }
                
                for category, count in speed_categories.items():
                    percentage = (count / total_non_zero) * 100
                    print(f"   {category}: {count} ({percentage:.1f}%)")
                
                # Print corrected ROS statistics
                print(f"\n   ROS STATISTICS (NON-ZERO speeds):")
                print(f"   Mean: {speed_mh.mean():.0f} m/h ({speed_mh.mean()/3600:.3f} m/s)")
                print(f"   Median: {speed_mh.median():.0f} m/h ({speed_mh.median()/3600:.3f} m/s)")
                print(f"   Max: {speed_mh.max():.0f} m/h ({speed_mh.max()/3600:.3f} m/s)")
                print(f"   Std Dev: {speed_mh.std():.0f} m/h ({speed_mh.std()/3600:.3f} m/s)")
            else:
                print("\n   WARNING: No non-zero speed records found!")
    
    return merged_df


def debug_propagation_analysis(byram_gdf, output_prefix):
    """
    Create debug visualizations to verify propagation calculation.
    Generates two GeoPackages: one with 90th percentile points and one with maximum points.
    """
    print("\n=== CREATING DEBUG VISUALIZATIONS ===")
    print("Generating two sets of propagation points for comparison:")
    print("1. 90th percentile (representative, robust)")
    print("2. Maximum distance (absolute, for reference)")
    
    debug_data_90th = []
    debug_data_max = []
    
    for fire_id in byram_gdf['fire_id'].unique():
        fire_data = byram_gdf[byram_gdf['fire_id'] == fire_id].copy()
        fire_data = fire_data.sort_values('datetime_hour')
        
        for idx, row in fire_data.iterrows():
            # 1. Save 90th percentile point (more representative)
            point_90th = row.get('point_90th_percentile', None)
            if point_90th is not None and hasattr(point_90th, 'x'):
                debug_data_90th.append({
                    'fire_id': fire_id,
                    'datetime_hour': row['datetime_hour'],
                    'datetime_display': row['datetime_display'],
                    'propagation_distance_90th_m': row['propagation_distance_m'],
                    'propagation_distance_max_m': row.get('propagation_distance_max_m', 0),
                    'propagation_speed_ms': row['propagation_speed_ms'],
                    'propagation_speed_kmh': row['propagation_speed_kmh'],
                    'radiative_intensity_kw_m': row.get('byram_radiative_intensity_kw_m', 0),
                    'traditional_intensity_kw_m': row.get('byram_traditional_intensity_kw_m', 0),
                    'fire_front_length_m': row.get('fire_front_length_m', 0),
                    'geometry': point_90th,
                    'point_type': '90th_percentile_representative',
                    'distance_reduction_pct': row.get('propagation_distance_reduction_pct', 0),
                    'n_samples': row.get('propagation_n_samples', 0)
                })
            
            # 2. Save maximum distance points
            point_max = row.get('point_maximum', None)
            if point_max is not None and hasattr(point_max, 'x'):
                debug_data_max.append({
                    'fire_id': fire_id,
                    'datetime_hour': row['datetime_hour'],
                    'datetime_display': row['datetime_display'],
                    'propagation_distance_90th_m': row['propagation_distance_m'],
                    'propagation_distance_max_m': row.get('propagation_distance_max_m', 0),
                    'propagation_speed_ms': row['propagation_speed_ms'],
                    'propagation_speed_kmh': row['propagation_speed_kmh'],
                    'radiative_intensity_kw_m': row.get('byram_radiative_intensity_kw_m', 0),
                    'traditional_intensity_kw_m': row.get('byram_traditional_intensity_kw_m', 0),
                    'fire_front_length_m': row.get('fire_front_length_m', 0),
                    'geometry': point_max,
                    'point_type': 'maximum_distance',
                    'distance_reduction_pct': row.get('propagation_distance_reduction_pct', 0),
                    'n_samples': row.get('propagation_n_samples', 0)
                })
    
    # Save 90th percentile points
    if debug_data_90th:
        debug_gdf_90th = gpd.GeoDataFrame(debug_data_90th, crs=byram_gdf.crs)
        debug_output_90th = f"{output_prefix}_propagation_vectors_90th_percentile.gpkg"
        debug_gdf_90th.to_file(debug_output_90th, driver='GPKG')
        
        print(f"\n   90th PERCENTILE PROPAGATION VECTORS:")
        print(f"   File: {debug_output_90th}")
        print(f"   Points created: {len(debug_gdf_90th)}")
        
        # Calculate average reduction
        avg_reduction = debug_gdf_90th['distance_reduction_pct'].mean() if len(debug_gdf_90th) > 0 else 0
        print(f"   Average reduction vs maximum: {avg_reduction:.1f}%")
    else:
        print("\n   No 90th percentile propagation points to save")
        debug_gdf_90th = None
    
    # Save maximum distance points
    if debug_data_max:
        debug_gdf_max = gpd.GeoDataFrame(debug_data_max, crs=byram_gdf.crs)
        debug_output_max = f"{output_prefix}_propagation_vectors_max.gpkg"
        debug_gdf_max.to_file(debug_output_max, driver='GPKG')
        
        print(f"\n   MAXIMUM DISTANCE PROPAGATION VECTORS:")
        print(f"   File: {debug_output_max}")
        print(f"   Points created: {len(debug_gdf_max)}")
        
        # Compare with 90th percentile
        if debug_gdf_90th is not None and len(debug_gdf_max) > 0:
            avg_distance_90th = debug_gdf_90th['propagation_distance_90th_m'].mean()
            avg_distance_max = debug_gdf_max['propagation_distance_max_m'].mean()
            
            if avg_distance_max > 0:
                reduction = (1 - avg_distance_90th / avg_distance_max) * 100
                print(f"   90th percentile vs maximum reduction: {reduction:.1f}%")
    else:
        print("\n   No maximum distance propagation points to save")
        debug_gdf_max = None
    
    # Print summary comparison
    if debug_gdf_90th is not None and debug_gdf_max is not None:
        print(f"\n   COMPARISON SUMMARY:")
        print(f"   90th percentile points: {len(debug_gdf_90th)}")
        print(f"   Maximum distance points: {len(debug_gdf_max)}")
        print(f"   Difference: {abs(len(debug_gdf_90th) - len(debug_gdf_max))} points")
        
        # Compare speeds
        avg_speed_90th = debug_gdf_90th['propagation_speed_ms'].mean()
        avg_speed_max = debug_gdf_max['propagation_speed_ms'].mean()
        
        if avg_speed_max > 0:
            speed_ratio = avg_speed_90th / avg_speed_max
            print(f"   Average speed ratio (90th/max): {speed_ratio:.2f}")
    
    return debug_gdf_90th, debug_gdf_max


def create_non_overlapping_byram_intensity(non_overlapping_gdf, byram_gdf, fire_id_column='fire_id'):
    """
    Create a non-overlapping progression GeoDataFrame with Byram intensity columns.
    This combines the non-overlapping polygons (new area each hour) with the Byram intensity metrics.
    """
    print("=== CREATING NON-OVERLAPPING BYRAM INTENSITY ===")
    
    # Ensure datetime_hour is timezone aware for both DataFrames
    for gdf in [non_overlapping_gdf, byram_gdf]:
        if 'datetime_hour' in gdf.columns and gdf['datetime_hour'].dt.tz is None:
            gdf['datetime_hour'] = gdf['datetime_hour'].dt.tz_localize('UTC')
    
    # Select ALL desired columns from byram_gdf and avoid duplication
    byram_columns = [
        'fire_id', 'datetime_hour',
        # Basic metrics
        'n_points_current_hour', 'frp_current_hour',
        'ratio_value', 'shrink_applied', 'shrink_distance_m',
        # Geometry and points
        'point_90th_percentile', 'point_maximum',
        'new_area_geometry',
        # Propagation metrics
        'propagation_distance_m', 'propagation_distance_max_m',
        'propagation_distance_reduction_pct',
        'propagation_speed_ms', 'propagation_speed_kmh',
        # Fire front length
        'fire_front_length_m',
        # FRP metrics
        'frp_new_area_mw', 'n_points_new_area',
        # Intensity
        'byram_radiative_intensity_kw_m', 
        'byram_traditional_intensity_kw_m',
        'radiative_efficiency',
        # Statistics
        'propagation_n_samples', 'propagation_mean_distance_m',
        'propagation_median_distance_m', 'propagation_std_distance_m',
        # Classification
        'intensity_class', 'traditional_intensity_class',
        # Validation
        'implied_radiative_fraction'
    ]
    
    # Check which columns are actually present in byram_gdf
    available_columns = [col for col in byram_columns if col in byram_gdf.columns]
    
    # Check which columns are missing
    important_columns = [
        'propagation_distance_m', 'propagation_distance_max_m',
        'propagation_distance_reduction_pct', 'fire_front_length_m',
        'byram_radiative_intensity_kw_m', 'byram_traditional_intensity_kw_m'
    ]
    
    missing_important = [col for col in important_columns if col not in available_columns]
    if missing_important:
        print(f"   WARNING: Missing important columns: {missing_important}")
    
    print(f"   Available columns from byram_gdf: {len(available_columns)}")
    
    # Check for column conflicts (geometry columns cannot be duplicated)
    conflict_columns = []
    for col in available_columns:
        if col in non_overlapping_gdf.columns and col != 'fire_id' and col != 'datetime_hour':
            conflict_columns.append(col)
    
    if conflict_columns:
        print(f"   WARNING: Column conflicts found: {conflict_columns}")
        print(f"   Will keep values from Byram data for these columns")
    
    # Merge the non-overlapping polygons with the Byram intensity data
    merged_gdf = non_overlapping_gdf.merge(
        byram_gdf[available_columns],
        on=['fire_id', 'datetime_hour'],
        how='left',
        suffixes=('', '_byram')
    )
    
    # Handle column conflicts by keeping Byram values
    for col in conflict_columns:
        if f'{col}_byram' in merged_gdf.columns:
            # Replace non-overlapping values with Byram values
            merged_gdf[col] = merged_gdf[f'{col}_byram']
            # Drop the duplicate column
            merged_gdf = merged_gdf.drop(columns=[f'{col}_byram'])
    
    # Fill missing values with 0 for numeric columns (if any)
    numeric_columns = merged_gdf.select_dtypes(include=[np.number]).columns
    merged_gdf[numeric_columns] = merged_gdf[numeric_columns].fillna(0)
    
    # Ensure datetime columns are properly formatted
    datetime_columns = ['datetime_hour', 'datetime_display']
    for col in datetime_columns:
        if col in merged_gdf.columns and merged_gdf[col].dt.tz is None:
            merged_gdf[col] = merged_gdf[col].dt.tz_localize('UTC')

    # Print summary of what was included
    print(f"Non-overlapping Byram intensity created: {len(merged_gdf)} polygons")
    
    # Count propagation metrics included
    propagation_metrics = ['propagation_distance_m', 'propagation_distance_max_m', 
                          'propagation_distance_reduction_pct']
    included_propagation = [col for col in propagation_metrics if col in merged_gdf.columns]
    
    print(f"   Propagation metrics included: {len(included_propagation)}/{len(propagation_metrics)}")
    if included_propagation:
        print(f"   Included: {included_propagation}")
    
    return merged_gdf


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
    
    # Non-overlapping progression option
    parser.add_argument('--no_non_overlapping', action='store_true', 
                   help='Disable non-overlapping progression (enabled by default)')
    
    # Calibration
    parser.add_argument('--reference_areas', help='Reference areas file for calibration')
    parser.add_argument('--skip_calibration', action='store_true', help='Skip calibration even if reference areas provided')
    
    # Byram intensity calculation
    parser.add_argument('--calculate_intensity', action='store_true',
                   help='Calculate Byram fire intensity (propagation speed and FRP in new area)')
    parser.add_argument('--radiative_fraction', type=float, default=0.15,
                   help='Radiative fraction for Byram calculation. Xr = 0.15-0.20 for wildfires (Wooster et al., 2005; Johnston et al., 2017) (default: 0.15)')
    parser.add_argument('--fuel_consumption', type=float, default=None,
                   help='Fixed fuel consumption in kg/m². If None, uses speed-adjusted values (default: None)')
    parser.add_argument('--fuel_type', type=str, default='shrub',
                   choices=['grass', 'shrub', 'forest'],
                   help='Fuel type for continuous consumption model (default: shrub)')
    parser.add_argument('--mtg_correction', type=float, default=0.6,
                   help='Correction factor for MTG 1km resolution artifacts. Typical values: 0.6-0.7. Lower values reduce L more aggressively (default: 0.6)')
    
    # Processing options
    parser.add_argument('--frp_threshold_method', choices=['fixed', 'adaptive'], default='adaptive', 
                       help='FRP threshold method (default: adaptive)')
    parser.add_argument('--frp_quantile_threshold', type=float, default=0.15,
                       help='Quantile for adaptive FRP threshold (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validation
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.min_area_non_overlapping < 0:
        print("ERROR: min_area_non_overlapping must be positive")
        sys.exit(1)
    
    print("\n=== FIRE PROGRESSION SYSTEM ===")
    
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
    process_non_overlapping = not args.no_non_overlapping
    
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
            print(f"\n   Initial cumulative progression saved: {output_file}")
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
            print(f"\n   Initial hourly progression saved: {output_file}")
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
                        print(f"\n   Calibrated cumulative progression saved: {output_file}")
                
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
                        print(f"\n   Calibrated hourly progression saved: {output_file}")
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
    
    
    # 5. CALCULATE IMPROVED BYRAM INTENSITY
    if hasattr(args, 'calculate_intensity') and args.calculate_intensity and process_cumulative:
        print("\n5. Calculating IMPROVED fire propagation speed and Byram intensity...")
        
        # Use calibrated version if available, otherwise initial
        if calibration_applied and 'cumulative_calibrated' in results:
            cumulative_gdf = results['cumulative_calibrated']
            calibrated_suffix = "_calibrated"
            byram_source = "cumulative_calibrated"
        else:
            cumulative_gdf = results['cumulative_initial']
            calibrated_suffix = "_initial"
            byram_source = "cumulative_initial"
        
        # 5.1 Calculate smart propagation speed using fire front vectors
        speed_df = calculate_smart_propagation_speed(cumulative_gdf, gdf, args.fire_id_column, mtg_correction=args.mtg_correction)
        
        # 5.2 Calculate FRP only in the new burned area (not cumulative)
        frp_new_area_df = calculate_frp_in_new_area(speed_df, gdf, args.fire_id_column)
        
        # 5.3 Calculate accurate Byram intensity with corrected inputs
        byram_gdf = calculate_byram_with_radiative_fraction(
            cumulative_gdf, 
            speed_df, 
            frp_new_area_df,
            radiative_fraction=args.radiative_fraction,
            fuel_consumption=args.fuel_consumption,
            fuel_type=args.fuel_type,
            mtg_correction=args.mtg_correction
        )
        
        # Store Byram results for later use
        results['byram_gdf'] = byram_gdf
        results['byram_suffix'] = calibrated_suffix
        results['byram_source'] = byram_source
        
        # Save the main results with intensity columns
        output_file = f"{args.output_prefix}_byram_improved{calibrated_suffix}.gpkg"
        byram_gdf.to_file(output_file, driver='GPKG')
        print(f"\n   Improved Byram intensity results saved: {output_file}")

        # Save CSV with detailed metrics for analysis
        csv_output = f"{args.output_prefix}_fire_metrics_improved{calibrated_suffix}.csv"

        # Define ALL possible columns we might want in the CSV
        csv_columns = [
            # Basic identification
            'fire_id', 'datetime_hour', 'datetime_display',
            # Area and FRP
            'area_ha', 'frp_current_hour', 'frp_new_area_mw', 'n_points_new_area',
            # Propagation metrics (90th percentile)
            'propagation_speed_ms', 'propagation_speed_kmh',
            'propagation_distance_m', 'propagation_distance_max_m',
            'propagation_distance_reduction_pct',
            # Propagation statistics
            'propagation_n_samples', 'propagation_mean_distance_m',
            'propagation_median_distance_m', 'propagation_std_distance_m',
            # Fire front length
            'fire_front_length_m',
            # Intensity
            'byram_radiative_intensity_kw_m', 'byram_traditional_intensity_kw_m',
            'radiative_efficiency', 'implied_radiative_fraction',
            # Classification
            'intensity_class', 'traditional_intensity_class',
            # Calc parameters
            'ratio_value', 'shrink_applied', 'shrink_distance_m',
            'adjusted_fuel_consumption_kg_m2'
        ]

        # Filter to only columns that exist in byram_gdf
        available_csv_columns = [col for col in csv_columns if col in byram_gdf.columns]

        # Add any additional numeric columns not in the list
        numeric_cols = byram_gdf.select_dtypes(include=[np.number]).columns
        additional_numeric = [col for col in numeric_cols if col not in available_csv_columns 
                              and col not in ['geometry', 'point_90th_percentile', 'point_maximum']]

        # Combine all columns
        all_csv_columns = available_csv_columns + list(additional_numeric)

        # Create CSV DataFrame
        metrics_df = byram_gdf[all_csv_columns].copy()

        # Save to CSV
        metrics_df.to_csv(csv_output, index=False)
        print(f"   Improved fire metrics CSV saved: {csv_output}")
        print(f"   Total columns: {len(all_csv_columns)}")
        print(f"   Key propagation metrics included: {'propagation_distance_max_m' in all_csv_columns}")
        print(f"   Distance reduction metric included: {'propagation_distance_reduction_pct' in all_csv_columns}")

        # Create debug visualizations to verify propagation vectors
        debug_gdf_90th, debug_gdf_max = debug_propagation_analysis(byram_gdf, args.output_prefix)
    
    # 6. CREATE NON-OVERLAPPING PROGRESSION
    if process_non_overlapping and process_cumulative:
        print("\n6. Creating non-overlapping progression...")
        
        # Use calibrated version if available, otherwise initial
        if calibration_applied and 'cumulative_calibrated' in results:
            cumulative_gdf = results['cumulative_calibrated']
            calibrated_suffix = "_calibrated"
        else:
            cumulative_gdf = results['cumulative_initial']
            calibrated_suffix = "_initial"
        
        # 6.1 Create hourly non-overlapping progression
        print("\n6.1 Creating hourly non-overlapping progression...")
        non_overlapping_hourly = create_non_overlapping_progression(
            cumulative_gdf,
            fire_id_column=args.fire_id_column,
            min_area_ha=args.min_area_non_overlapping
        )
        
        if non_overlapping_hourly is not None:
            results['non_overlapping_hourly'] = non_overlapping_hourly
            output_file = f"{args.output_prefix}_non_overlapping_hourly{calibrated_suffix}.gpkg"
            non_overlapping_hourly.to_file(output_file, driver='GPKG')
            print(f"\n   Non-overlapping hourly progression saved: {output_file}")
            
            # 6.2 Create daily non-overlapping summary
            print("\n6.2 Creating daily non-overlapping summary...")
            daily_non_overlapping = create_daily_non_overlapping(
                non_overlapping_hourly,
                min_area_ha=args.min_area_non_overlapping
            )
            
            if daily_non_overlapping is not None:
                results['daily_non_overlapping'] = daily_non_overlapping
                output_file = f"{args.output_prefix}_daily_non_overlapping{calibrated_suffix}.gpkg"
                daily_non_overlapping.to_file(output_file, driver='GPKG')
                print(f"\n   Daily non-overlapping summary saved: {output_file}")
            
            # 6.3 Create non-overlapping Byram intensity version
            if hasattr(args, 'calculate_intensity') and args.calculate_intensity and 'byram_gdf' in results:
                print("\n6.3 Creating non-overlapping Byram intensity...")
                non_overlapping_byram = create_non_overlapping_byram_intensity(
                    non_overlapping_hourly, 
                    results['byram_gdf'],
                    fire_id_column=args.fire_id_column
                )
                
                if non_overlapping_byram is not None:
                    results['non_overlapping_byram'] = non_overlapping_byram
                    output_file = f"{args.output_prefix}_non_overlapping_hourly_byram_improved{calibrated_suffix}.gpkg"
                    non_overlapping_byram.to_file(output_file, driver='GPKG')
                    print(f"\n   Non-overlapping Byram intensity saved: {output_file}")
        else:
            print("   ERROR: No non-overlapping polygons were created")
    
    # 7. CREATE DAILY SUMMARIES FOR ALL VERSIONS
    print("\n7. Creating daily summaries...")
    
    if process_cumulative and 'cumulative_initial' in results:
        daily_summary_cumulative = create_daily_summary(results['cumulative_initial'], progression_type="cumulative")
        if calibration_applied and 'cumulative_calibrated' in results:
            # Use the calibrated version for output
            daily_summary_cumulative_calibrated = create_daily_summary(
                results['cumulative_calibrated'], progression_type="cumulative_calibrated"
            )
            output_file = f"{args.output_prefix}_daily_cumulative_calibrated.gpkg"
            daily_summary_cumulative_calibrated.to_file(output_file, driver='GPKG')
            print(f"   Calibrated cumulative daily summary saved: {output_file}")
        else:
            output_file = f"{args.output_prefix}_daily_cumulative_initial.gpkg"
            daily_summary_cumulative.to_file(output_file, driver='GPKG')
            print(f"   Initial cumulative daily summary saved: {output_file}")
    
    if process_hourly and 'hourly_initial' in results:
        daily_summary_hourly = create_daily_summary(results['hourly_initial'], progression_type="hourly")
        if calibration_applied and 'hourly_calibrated' in results:
            # Use the calibrated version for output
            daily_summary_hourly_calibrated = create_daily_summary(
                results['hourly_calibrated'], progression_type="hourly_calibrated"
            )
            output_file = f"{args.output_prefix}_daily_hourly_calibrated.gpkg"
            daily_summary_hourly_calibrated.to_file(output_file, driver='GPKG')
            print(f"   Calibrated hourly daily summary saved: {output_file}")
        else:
            output_file = f"{args.output_prefix}_daily_hourly_initial.gpkg"
            daily_summary_hourly.to_file(output_file, driver='GPKG')
            print(f"   Initial hourly daily summary saved: {output_file}")
    
    # 8. FINAL STATISTICS
    print("\n8. FINAL STATISTICS:")
    print(f"   Total fires processed: {gdf[args.fire_id_column].nunique()}")
    print(f"   Shrink_distance applied: {calibration_shrink} meters")
    print(f"   Calibration applied: {'YES' if calibration_applied else 'NO'}")
    print(f"   Non-overlapping progression: {'YES (default)' if process_non_overlapping else 'NO (disabled)'}")
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


    # 9. IMPORTANT NOTES AND CLARIFICATIONS
    print("\n" + "="*70)
    print("IMPORTANT NOTES AND CLARIFICATIONS")
    print("="*70)

    print(f"\n1. PROPAGATION METRICS (key change in this version):")
    print(f"   ------------------------------------------------")
    print(f"   • propagation_distance_m: 90th percentile distance (used in calculations)")
    print(f"   • propagation_distance_max_m: Maximum absolute distance (for reference)")
    print(f"   • propagation_distance_reduction_pct: % reduction (90th vs max)")
    print(f"   ")
    print(f"   Why this change?")
    print(f"   - MTG 1km resolution can create 'spiky' propagation patterns")
    print(f"   - Maximum distance often represents a single anomalous pixel")
    print(f"   - 90th percentile gives more representative propagation")
    print(f"   - Reduction_pct > 30% suggests data quality issues")

    print(f"\n2. INTENSITY CALCULATIONS:")
    print(f"   ------------------------------------------------")
    print(f"   • byram_radiative_intensity_kw_m: FRP-based (I = FRP / (L × Xr))")
    print(f"   • byram_traditional_intensity_kw_m: Fuel-based (I = H × w × v)")
    print(f"   • radiative_efficiency: Ratio between the two methods")
    print(f"   • implied_radiative_fraction: What Xr would equalize both methods")

    print(f"\n3. FIRE FRONT LENGTH (L):")
    print(f"   ------------------------------------------------")
    print(f"   • Calculated with MTG correction factor: {args.mtg_correction}")
    print(f"   • Accounts for 1km resolution geometric artifacts")
    print(f"   • Real front length ≈ Calculated length × {args.mtg_correction}")

    print(f"\n4. NON-OVERLAPPING POLYGONS:")
    print(f"   ------------------------------------------------")
    print(f"   • Each polygon shows ONLY new burned area for that hour")
    print(f"   • Useful for print layouts and progression visualization")
    print(f"   • Minimum area filter: {args.min_area_non_overlapping} ha")
    print("="*70)


if __name__ == "__main__":
    main()