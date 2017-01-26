%% Build supervised dataset
% Comment: Combine measurements from vehicles and weather station.
% The dataset is labeled and stores measurements each hour from the 
% last four hours (by default)
% Important variables with deafult values:
%
% Step size (in minutes)
% inc_min_step = 2
% Duration until last measurement
% prev_time = 60
% Set offset, this defines the forecast model (30 min, 60 min, 90 min...)
% offset_time = loop
% Set minimum quality level
% min_quality = 4
% Set maximum duration since last measurement in hours
% max_search_time = 5
% Set search region (for prev. measured friction values)
% search_region = 0.04

%% For extracting different setups
for loop = 120:30:120


%%
%clear
%% Load data from the vehicles
% These files are generated by a sql query
%SELECT datanew.id, data.sessionid, data.receivetime, 
%   data.detectiontype, Format([frictionvalue],"0.00000") AS frictionvalue_, 
%   data.frictionquality, Format([ambienttemperature],"0.00000") AS ambienttemperature_, 
%   data.wiperspeed, Format([latitude],"0.00000") AS latitude_,
%   Format([longitude],"0.00000") AS longitude_, 
%   Format([mappedlatitude],"0.00000") AS mappedlatitude_, 
%   Format([mappedlongitude],"0.00000") AS mappedlongitude_, data.segment
%FROM data
%WHERE data.segment=833373149;

fid = fopen('../export/Query_748861108_748861109_or_748861110_or_748861111_new.csv', 'rt');
raw = textscan(fid, ['%s %s %s',repmat('%f',[1,10])],'Delimiter',',','headerLines', 0);
fclose(fid);

%% Define date/time format
formatIn = 'yyyy-mm-dd HH:MM:SS';
formatIn2 = 'dd-mmm-yyyy HH:MM:SS';
formatIn3 = 'mm/dd/yyyy HH:MM';

% Set start and end date/time
starttime = '2015-11-01 14:44:06';
endtime = '2016-11-15 06:06:07';

%Step size (in minutes)
inc_min_step = 2;
% Duration until last measurement (default 60 min)
prev_time = 60;
% Set offset, this defines the forecast model (30 min, 60 min, 90 min...)
offset_time = loop;
% Set minimum quality level
min_quality = 4;
% Set maximum duration since last measurement in hours (default 5 hours)
max_search_time = 5;
% Set search region (for prev. measured friction values) (defalt 0.04, gps
% coords
search_region = 0.04;

% Error check
if inc_min_step > prev_time
    error('Cannot handle prev_time time larger then inc_step')
end

% Get the start and end dates in serial date number
step_behind = prev_time/inc_min_step;
firsttimemark = datenum(starttime,formatIn);
current = datenum(starttime,formatIn);
lasttime = datenum(endtime,formatIn);

% Show start and end date/time
disp('Start time')
datestr(current,formatIn)
disp('End time')
datestr(lasttime,formatIn)


t1 = datevec(firsttimemark);
t2 = datevec(lasttime);
% Get number of time intervals from start to end date/time
nummins = etime(t2,t1)/60/inc_min_step;

%% Feature Index
% These are the indices for the different features
indFrictionValue = 3;
indFrictionQuality = 4;

ind1PrevDistFriction = 5;
ind1PrevTimeFriction = 6;
ind1PrevFrictionValue = 7;
ind1PrevFrictionQuality = 65;

indTempSMHI = 9;
indTempRoadVV = 14;
indTempVV = 19;
indHumidityVV = 24;
indDewVV = 29;
indRainVV = 34;
indSnowVV = 39;
indWindVV = 44;
indWiperSpeedCar = 49;
indLog = 54;
indLat = 55;
indMappedLog = 56;
indMappedLat = 57;

ind2PrevDistFriction = 58;
ind2PrevTimeFriction = 59;
ind2PrevFrictionValue = 60;
ind3PrevDistFriction = 61;
ind3PrevTimeFriction = 62;
ind3PrevFrictionValue = 63;
indTempCar = 64;


%% Fix the temperature from car data (not used)
% Fill in the gaps in the dataset. 
lastTemp = mean(raw{7}(~isnan(raw{7}(:))));
for i=1:length(raw{7}(:))
    if ~isnan(raw{7}(i))
        lastTemp = raw{7}(i);
    end
    raw{7}(i) = lastTemp;
end


%% Fix wiperspeed values (not used)
% Fill in the gaps in the dataset. 
lastTemp = 0;
for i = 1:length(raw{8})
    if(~isnan(raw{8}(i)))
        lastTemp = raw{8}(i);
   end
   raw{8}(i) = lastTemp;
end

%% Load friction values from car data...
disp('Load friction values from car data...')
% Allocate some space for the dataset
dataset = zeros(fix(nummins),64);
% Get start date
current = datenum(starttime,formatIn);
% Dummy variable to optimize the search algorithm
lowestIndex=1;
% Loop through every time interval
for mins=1:nummins
    
    % Progress output
    if mod(mins,1000) == 0
        fprintf('%f %% done\n',mins/nummins*100);
    end
    current_end = addtodate(current, inc_min_step, 'minute');
    
    % Set an offset, used to build datasets for the forecast models
    current_offset = addtodate(current, -offset_time, 'minute');
    current_end_offset = addtodate(current_end, -offset_time, 'minute');
    
    % Dummy variables used to calculate the average
    numfriction = 0;
    numwiperspeedvalues = 0;
    numtempvalues=0;
    
    for m=lowestIndex:length(raw{3}(:))
        
        % Find matching measurement
        if (datenum(raw{3}(m),formatIn) >= current) && ...
           (datenum(raw{3}(m),formatIn) < current_end) && ...
           ~isnan(raw{5}(m)) && ...
           (raw{6}(m) >= min_quality)
       
           % Save index to dataset
           dataset(mins,2) = m;
           
           % Store friction value and quality value
           dataset(mins,indFrictionValue) = dataset(mins,indFrictionValue)+raw{5}(m);
           dataset(mins,indFrictionQuality) = dataset(mins,indFrictionQuality)+raw{6}(m);
           
           % Save log/lat and mapped log/lat
           dataset(mins,indLog) = raw{9}(m);
           dataset(mins,indLat) = raw{10}(m);
           dataset(mins,indMappedLog) = raw{11}(m);
           dataset(mins,indMappedLat) = raw{12}(m);

           
           numfriction = numfriction + 1;
        end
        
        % Looking for wiperspeed value
        if (datenum(raw{3}(m),formatIn) >= current_offset) && ...
           (datenum(raw{3}(m),formatIn) < current_end_offset)
           dataset(mins,indWiperSpeedCar) = dataset(mins,indWiperSpeedCar)+raw{8}(m);
           numwiperspeedvalues = numwiperspeedvalues + 1;
        end
        
        % Look for temperature values
        if (datenum(raw{3}(m),formatIn) >= current_offset) && ...
           (datenum(raw{3}(m),formatIn) < current_end_offset)
            dataset(mins,indTempCar) = dataset(mins,indTempCar)+raw{7}(m);
            numtempvalues = numtempvalues + 1;
        end
        
        % Optimize the search algorithm
        if datenum(raw{3}(m),formatIn) < current_offset
            lowestIndex = m;
        end
        if datenum(raw{3}(m),formatIn) > current_end
            dataset(mins,2) = m;
            break
        end
    end
    
    if dataset(mins,indFrictionValue) ~= 0 % Set mean of friction value
        dataset(mins,indFrictionValue) = dataset(mins,indFrictionValue)/numfriction;
        dataset(mins,indFrictionQuality) = dataset(mins,indFrictionQuality)/numfriction;
    end
    
    if dataset(mins,indWiperSpeedCar) ~= 0 % Set mean of wiper speed
        dataset(mins,indWiperSpeedCar) = dataset(mins,indWiperSpeedCar)/numwiperspeedvalues;
    end
    if dataset(mins,indTempCar) ~= 0 % Set mean of temperature
        dataset(mins,indTempCar) = dataset(mins,indTempCar)/numtempvalues;
    end
    
    % Store the current time
    dataset(mins,1) = current;
    
    % Save time stamp
    [~,~,~,dataset(mins,8),~,~] = datevec(datestr(current),formatIn2);
    
    % Update the time
    current = current_end;
end

hold off

% Copy dataset into newdataset
newdataset = dataset;


%% Find the middle of the road segment (Not used)
middleLog = mean(dataset((dataset(:,indMappedLog) > 0),indMappedLog));
middleLat = mean(dataset((dataset(:,indMappedLat) > 0),indMappedLat));


%% LOAD data from all friction values
fid = fopen('../export/GetAllFrictionValues.csv', 'rt'); %Query5_onlyfriction.csv
raw_all = textscan(fid, ['%s %s %s',repmat('%f',[1,10])],'Delimiter',',','headerLines', 0);
fclose(fid);

% Allocate some space
alldataset = zeros(fix(nummins),50);

disp('Load friction values from all friction values...')
current = datenum(starttime,formatIn);

% Dummy optimizer variable
lowestIndex=1;

% Loop through all time intervals
for mins=1:nummins
    
    % Progress output
    if mod(mins,40) == 0
        fprintf('%f %% done\n',mins/nummins*100);
    end
    current_end = addtodate(current, inc_min_step, 'minute');
    
    % Set an offset, used to build datasets for the forecast models
    current_offset = addtodate(current, -offset_time, 'minute');
    current_end_offset = addtodate(current_end, -offset_time, 'minute');
    
    numfriction = 0;
    
    % Loop through all friction measurements
    for m=lowestIndex:length(raw_all{3}(:))
        
        % Find matching measurement
        if (datenum(raw_all{3}(m),formatIn) >= current_offset) && ...
           (datenum(raw_all{3}(m),formatIn) < current_end_offset) && ...
           (raw_all{6}(m) >= min_quality)
       
           alldataset(mins,indFrictionValue) = alldataset(mins,indFrictionValue)+raw_all{5}(m);
           alldataset(mins,indFrictionQuality) = alldataset(mins,indFrictionQuality)+raw_all{6}(m);
           
           alldataset(mins,49) = raw_all{9}(m);
           alldataset(mins,50) = raw_all{10}(m);
           
           numfriction = numfriction + 1;
        end
        
        % Optimize the search algorithm
        if datenum(raw_all{3}(m),formatIn) < current_offset
            lowestIndex = m;
        end
        if datenum(raw_all{3}(m),formatIn) > current_end_offset
            break
        end
    end
    
    if alldataset(mins,indFrictionValue) ~= 0 % Set mean friction value and quality
        alldataset(mins,indFrictionValue) = alldataset(mins,indFrictionValue)/numfriction;
        alldataset(mins,indFrictionQuality) = alldataset(mins,indFrictionQuality)/numfriction;
    end
    
    % Save the current time
    alldataset(mins,1) = current;

    % Update the time mark
    current = current_end;
end

%%  FIND LAST GLOBAL FRICTIONVALUE (TEST CASE)
disp('Look for global friction values')

% Loop through all time intervals
for mins=1:nummins
    
    % Progress output
    if mod(mins,40) == 0
        fprintf('%f %% done\n',mins/nummins*100);
    end
    
    found_frictionvalue = false;
    
    for searchhour = mins-1:-1:2
        
        if (mins-searchhour)*inc_min_step > 5*60
            break;
        end
        
        % Find matching measurement
        if (alldataset(searchhour,indFrictionValue) ~= 0) && ...
           (sqrt((newdataset(mins,indMappedLog)-alldataset(searchhour,49))^2+...
           (newdataset(mins,indMappedLat)-alldataset(searchhour,50))^2) < search_region) && ...
           ((mins-searchhour)*inc_min_step < max_search_time*60) && ... % Max 5 hours
           (alldataset(searchhour,indFrictionQuality) >= min_quality) % Qulity needs to be better or equal to min_quality
       
            newdataset(mins,ind1PrevFrictionValue) = alldataset(searchhour,indFrictionValue);
            newdataset(mins,ind1PrevFrictionQuality) = alldataset(searchhour,indFrictionQuality);
            newdataset(mins,ind1PrevTimeFriction) = mins-searchhour;
            if alldataset(searchhour,49) + alldataset(searchhour,50) > 0
                newdataset(mins,ind1PrevDistFriction) = sqrt((newdataset(mins,indMappedLog)-alldataset(searchhour,49))^2+...
                    (newdataset(mins,indMappedLat)-alldataset(searchhour,50))^2);
            else
                newdataset(mins,ind1PrevDistFriction) = 0;
            end
            found_frictionvalue = true;
            break;
        end
    end
    if found_frictionvalue == false
        newdataset(mins,ind1PrevDistFriction) = 0;
        newdataset(mins,ind1PrevTimeFriction) = 0;
        newdataset(mins,ind1PrevFrictionValue) = 0;
    end
    
    
    found_frictionvalue = false;
    for searchhour = searchhour-1:-1:2
        if (alldataset(searchhour,indFrictionValue) ~= 0) && ...
            (sqrt((newdataset(mins,indMappedLog)-alldataset(searchhour,49))^2+...
           (newdataset(mins,indMappedLat)-alldataset(searchhour,50))^2) < search_region) && ...
           ((mins-searchhour)*inc_min_step < max_search_time*60) && ... % Max 5 hours
           (alldataset(searchhour,indFrictionQuality) >= min_quality) % Qulity needs to be better or equal to min_quality
            
            newdataset(mins,ind2PrevFrictionValue) = alldataset(searchhour,indFrictionValue);
            newdataset(mins,ind2PrevTimeFriction) = mins-searchhour;
            if alldataset(searchhour,49) + alldataset(searchhour,50) > 0
                newdataset(mins,ind2PrevDistFriction) = sqrt((newdataset(mins,indMappedLog)-alldataset(searchhour,49))^2+...
                    (newdataset(mins,indMappedLat)-alldataset(searchhour,50))^2);
            else
                newdataset(mins,ind2PrevDistFriction) = 0;
            end
            found_frictionvalue = true;
            break;
        end
    end
    

    
    if found_frictionvalue == false
        newdataset(mins,ind2PrevDistFriction) = 0;
        newdataset(mins,ind2PrevTimeFriction) = 2;
        newdataset(mins,ind2PrevFrictionValue) = 0.5;
    end
    
    
    found_frictionvalue = false;
    for searchhour = searchhour-1:-1:2
        if (alldataset(searchhour,indFrictionValue) ~= 0) && ...
            (sqrt((newdataset(mins,indMappedLog)-alldataset(searchhour,49))^2+...
           (newdataset(mins,indMappedLat)-alldataset(searchhour,50))^2) < search_region) && ...
           ((mins-searchhour)*inc_min_step < max_search_time*60) && ... % Max 5 hours
           (alldataset(searchhour,indFrictionQuality) >= min_quality) % Qulity needs to be better or equal to min_quality
       
            newdataset(mins,ind3PrevFrictionValue) = alldataset(searchhour,indFrictionValue);
            newdataset(mins,ind3PrevTimeFriction) = mins-searchhour;
            if alldataset(searchhour,49) + alldataset(searchhour,50) > 0
                newdataset(mins,ind3PrevDistFriction) = sqrt((newdataset(mins,indMappedLog)-alldataset(searchhour,49))^2+...
                    (newdataset(mins,indMappedLat)-alldataset(searchhour,50))^2);
            else
                newdataset(mins,ind3PrevDistFriction) = 0;
            end
            found_frictionvalue = true;
            break;
        end
    end
    
    if found_frictionvalue == false
        newdataset(mins,ind3PrevDistFriction) = 0;
        newdataset(mins,ind3PrevTimeFriction) = 2;
        newdataset(mins,ind3PrevFrictionValue) = 0.5;
    end
end

% Clear every friction value when distance is further then 10 (not used)
%newdataset(newdataset(:,ind1PrevDistFriction) > 10,ind1PrevDistFriction) = 0;
%newdataset(newdataset(:,ind2PrevDistFriction) > 10,ind2PrevDistFriction) = 0;
%newdataset(newdataset(:,ind3PrevDistFriction) > 10,ind3PrevDistFriction) = 0;
% Clear every friction value from 5 hours ago
%newdataset(newdataset(:,ind1PrevTimeFriction) > 300,ind1PrevTimeFriction) = 0;
%newdataset(newdataset(:,ind2PrevTimeFriction) > 300,ind2PrevTimeFriction) = 0;
%newdataset(newdataset(:,ind3PrevTimeFriction) > 300,ind3PrevTimeFriction) = 0;

%% ADD Temperature data from SMHI
ftemp = fopen('../export/SMHITemp.csv','rt');
raw_temp = textscan(ftemp, ['%f %s %s ',repmat('%f',[1,1])],'Delimiter',',','headerLines', 0); %or whatever formatting your file is
fclose(ftemp);

%% Create datetime list in raw_temp
% Reformat the time stamp
disp('Create datetime list in raw_temp...')
numtemphours = length(raw_temp{1}(:));
for temphours=1:numtemphours
    if mod(temphours,100) == 0
        fprintf('%.2f %% done\n',temphours/numtemphours*100);
    end
    raw_temp{5}(temphours) = datenum([...
        strjoin(raw_temp{2}(temphours)) ' ' strjoin(raw_temp{3}(temphours))],...
        formatIn);
end

%% Load data from SMHI
disp('Load temperatures from SMHI dataset...')
% Dummy optimizer variable
lowestIndex = 2;

% Loop through all time intervals
for mins=1:nummins
    
    % Progress output
    if mod(mins,10) == 0
        fprintf('%.2f %% done\n',mins/nummins*100);
    end
    
    % Set an offset, used to build datasets for the forecast models
    current_offset = addtodate(newdataset(mins,1), -offset_time, 'minute');
    datestr(newdataset(mins,1));
    for temphours=lowestIndex:numtemphours-1
        
        % Optimize the search algorithm
        if current_offset < raw_temp{5}(temphours)
            break
        end
        if raw_temp{5}(temphours) < current_offset
            lowestIndex = temphours;
        end
        
        % Find matching measurement
        if (current_offset >= raw_temp{5}(temphours)) && ...
           (current_offset < raw_temp{5}(temphours+1))
            newdataset(mins,indTempSMHI) = raw_temp{4}(temphours);
        end
    end
end

 
%% Store previous temperatures (SMHI)
 newdataset(:,indTempSMHI+1) = [zeros(step_behind*1,1);newdataset(1:end-step_behind*1,indTempSMHI)];
 newdataset(:,indTempSMHI+2) = [zeros(step_behind*2,1);newdataset(1:end-step_behind*2,indTempSMHI)];
 newdataset(:,indTempSMHI+3) = [zeros(step_behind*3,1);newdataset(1:end-step_behind*3,indTempSMHI)];
 newdataset(:,indTempSMHI+4) = [zeros(step_behind*4,1);newdataset(1:end-step_behind*4,indTempSMHI)];


%% ADD data from Vagverket
fvv = fopen('../export/query_weatherstation_save_1435.csv','rt');
raw_vv = textscan(fvv, ['%f %f %s ',repmat('%f',[1,8]),' %s %f'],'Delimiter',',','headerLines', 0); %or whatever formatting your file is
fclose(fvv);

% Get number of time intervals
numvvhours = length(raw_vv{1}(:));

%% Load datenum info array col 1
for vvhours=1:numvvhours
    raw_vv{1}(vvhours) = datenum(raw_vv{3}(vvhours),formatIn3);
end


%% LOAD data from vagverket
disp('Load data from vagverket...')
lowestIndex = 2;
for mins=1:nummins
    
    % Progress output
    if mod(mins,10) == 0
        fprintf('%.2f %% done\n',mins/nummins*100);
    end
    
    % Set an offset, used to build datasets for the forecast models
    current_offset = addtodate(newdataset(mins,1), -offset_time, 'minute');
    
    for vvhours=lowestIndex:numvvhours-1

        % Optimize the search algorithm
        if current_offset < raw_vv{1}(vvhours)
            break
        end
        if raw_vv{1}(vvhours) < current_offset
            lowestIndex = vvhours;
        end
        
        % Find matching measurement
        if current_offset >= raw_vv{1}(vvhours) &&...
           current_offset < raw_vv{1}(vvhours+1)
       
            % Add Road heat
            newdataset(mins,indTempRoadVV) = raw_vv{4}(vvhours);
            % Add Air temperature
            newdataset(mins,indTempVV) = raw_vv{5}(vvhours);
            % Add Air Humidity
            newdataset(mins,indHumidityVV) = raw_vv{6}(vvhours);
            % Add Daggpunktstemperatur
            newdataset(mins,indDewVV) = raw_vv{7}(vvhours);
            
            % Add Regn & Snow
            if raw_vv{8}(vvhours) == 1
                newdataset(mins,indRainVV) = 0;
                newdataset(mins,indSnowVV) = 0;
            elseif raw_vv{8}(vvhours) == 2 || ...
                    raw_vv{8}(vvhours) == 3
                newdataset(mins,indRainVV) = raw_vv{9}(vvhours);
                newdataset(mins,indSnowVV) = 0;
            elseif raw_vv{8}(vvhours) == 4
                newdataset(mins,indRainVV) = 0;
                newdataset(mins,indSnowVV) = raw_vv{9}(vvhours);
            elseif raw_vv{8}(vvhours) == 6
                newdataset(mins,indRainVV) = raw_vv{9}(vvhours);
                newdataset(mins,indSnowVV) = raw_vv{9}(vvhours);
            else
                newdataset(mins,indRainVV) = 0;
                newdataset(mins,indSnowVV) = 0;
            end

            % Add Wind
            newdataset(mins,indWindVV) = raw_vv{13}(vvhours);
        end
    end
end


%% Clear data from v�gverket
% Limit temperature
newdataset(newdataset(:,indTempRoadVV) < -30,indTempRoadVV) = mean(newdataset(:,indTempRoadVV));
% Limit humidity
newdataset(newdataset(:,indHumidityVV) < -30,indHumidityVV) = mean(newdataset(:,indHumidityVV));
% Cap Rain and Snow lvl at zero
newdataset(newdataset(:,indRainVV) < 0,indRainVV) = 0;
newdataset(newdataset(:,indSnowVV) < 0,indSnowVV) = 0;


%% Get previous measurements from...
% Road heat
% Airtemp
% Humidity
% Dewpoint temperature
% Rain
% Snow
% Wind speed
% Wiperspeed
counter = 15;
while counter < 53
    newdataset(:,counter) = [zeros(step_behind*1,1);newdataset(1:end-step_behind*1,counter-1)];
    counter = counter + 1;
    newdataset(:,counter) = [zeros(step_behind*2,1);newdataset(1:end-step_behind*2,counter-2)];
    counter = counter + 1;
    newdataset(:,counter) = [zeros(step_behind*3,1);newdataset(1:end-step_behind*3,counter-3)];
    counter = counter + 1;
    newdataset(:,counter) = [zeros(step_behind*4,1);newdataset(1:end-step_behind*4,counter-4)];
    counter = counter + 2;
end



%% Remove datapoint if there is no friction value
% Copy newdataset
cleareddataset = newdataset;

% Get serial date number from start and end date
current = datenum(starttime,formatIn);
lasttime = datenum(endtime,formatIn);

disp('remove datapoints without friction value...')
cleareddataset((cleareddataset(:,indFrictionValue) == 0),:) = [];
cleareddataset((cleareddataset(:,ind1PrevFrictionValue) == 0),:) = [];

%% Plot cleared dataset (for debugging)
% hold on
% 
% plot(cleareddataset(:,1),cleareddataset(:,indFrictionValue),'o')
% datetick('x','yyyy-mm-dd','keepticks')
% 
% hold off


%% Prep data for plotting (not used)
removelowerpointsnot = 0;
tempnewdataset = newdataset;
tempalldataset = alldataset;
newdataset(newdataset(:,indFrictionValue)==0,indFrictionValue) = -10;
alldataset(alldataset(:,indFrictionValue)==0,indFrictionValue) = -10;

if removelowerpointsnot == 1
    newdataset = tempnewdataset;
    alldataset = tempalldataset;
end

%% Save cleareddataset as .csv and .mat
csvwrite(['ANN/cleareddataset' num2str(loop) '.csv'],cleareddataset)
save(['cleareddataset' num2str(loop) '.mat'])


end



%%---------------------------------------
% For report (not used)
%----------------------------------------
% Plot friction and temp (for report and debugging)
%%%%%%%%%%hold on

%plot(newdataset(:,1),newdataset(:,indFrictionValue),'*r','markersize',10)
%plot(newdataset(:,1),alldataset(:,indFrictionValue),'omagenta')
%%%%%%%%%%%%plot(newdataset(:,1),newdataset(:,ind1PrevFrictionValue),'ored')

% Plot temp from VV
% s = normc(newdataset(:,19))
% norms = s - min(s(:))
% norms = norms ./ max(norms(:))
%plot(newdataset(:,1),norms,'--g')

% Plot temp from SMHI
% s = normc(newdataset(:,9))
% norms = s - min(s(:))
% norms = norms ./ max(norms(:))
%plot(newdataset(:,1),norms,'--y')

% Plot humidity
%%%%%%s = normc(newdataset(:,24));
%%%%%%norms = s - min(s(:));
%%%%%%norms = norms ./ max(norms(:));
%%%%%%%%plot(newdataset(:,1),norms,'--black')

% Plot regn
%%%%%%%%%%%plot(newdataset(:,1),newdataset(:,indRainVV),'-b')
%%%%%%datetick('x','yyyy-mm-dd','keepticks')

% Plot snow
%%%%%%%%%%5plot(newdataset(:,1),newdataset(:,indSnowVV),'color','[0.3 0.3 1.0]')
%datetick('x','yyyy-mm-dd','keepticks')
%set(gca, 'XTick', newdataset(:,1:6:end));
%%%%%%%%%%%tickDates = newdataset(1,1):1:newdataset(end,1); %// creates a vector of tick positions
%%%%%%%%%%%%set(gca, 'XTick' , tickDates , 'XTickLabel' , datestr(tickDates,'yyyy-mm-dd') )
%xticklabel_rotate;
% NumTicks = 300;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks))
% set(gca,'XMinorTick','on','YMinorTick','on')
% Plot wiperspeed
%plot(newdataset(:,1),newdataset(:,49),'-c')
%%%%%%hold off
%legend('Friction values (Segment)','Friciton values (Region)','SMHI Temperature','VV Temperature,','Humidity','Rainfall')
%%%%%%legend('Friction values','Humidity,','Rain')



%% Plot temperatures (for report)
%%%%%%%%hold off

% Plot temp from VV
%%%%%%%%%%%%%%plot(newdataset(:,1),newdataset(:,19),'-g')
%%%%%%%%hold on

% Plot temp from SMHI
%%%%%%%%%plot(newdataset(:,1),newdataset(:,9),'-b')


% Plot temp from car data
%%%%%%%%%newdataset(newdataset(:,indTempCar) == 0,indTempCar) = -100;
%%%%%%%%%%%%%plot(newdataset(:,1),newdataset(:,indTempCar),'or')
%%%%%%%%ylabel('Temperature')
%%%%%%%%legend('V�gverket','SMHI','From vehicles in the region')
%%%%%%%%datetick('x','yyyy-mm-dd','keepticks')
%xticklabel_rotate;
%%%%%%%%%%%%%%%tickDates = newdataset(1,1):1:newdataset(end,1); %// creates a vector of tick positions
%%%%%%%%%%%%%set(gca, 'XTick' , tickDates , 'XTickLabel' , datestr(tickDates,'yyyy-mm-dd') );
