function dataout=loadsegy(fname,traces,select);

% lettura file in formato Seg-Y
%
% dataout = loadsegy(fname,traces,selection)
%   dataout: seismic data output structure
%     fname: segy file name and path
%            (fdir='*' ask for input file)
%    traces: starting and ending trace (optional)
% selection: 'reel_head+tr_head+tr_data+tr_minmax+geometry+binning' (optional)
%          : 'allhead' = 'reel_head+tr_head+tr_minmax'
%          : 'allgeo'  = 'geometry+binning+tr_minmax'
%          : 'all'     = 'reel_head+tr_head+tr_data+tr_minmax+geometry+binning'

if (nargin==0 | fname=='*'),
  [fname,fdir]=uigetfile('*.s*y', 'Load segy data');
  if (isequal(fname,0)|isequal(fdir,0)), error('SEG-Y Loading aborted...');
  else,         fname=fullfile(fdir,fname);
  end,
end,

if (nargin<2), traces=[]; end,
if (nargin<3), select='reel_head+tr_head+tr_data'; end,
if (nargin==2 & ischar(traces)) select=traces; traces=[]; end,
if     (strcmp(select,'allhead')==1), select='reel_head+tr_head+tr_minmax';
elseif (strcmp(select,'allgeo')==1),  select='geometry+binning+tr_minmax';
elseif (strcmp(select,'all')==1),     select='reel_head+tr_head+tr_data+tr_minmax+geometry+binning'; 
end,

tr_h=0; tr_d=0; tr_m=0; tr_g=0; tr_b=0;

fid=fopen(fname,'r','ieee-be');
if (fid==-1)  error('file not found');  end,

fseek(fid,0,1); file_len=ftell(fid); fseek(fid,0,-1);

% load reel textual header
% ------------------------
if (~isempty(findstr(select,'reel_head')) ),
  dataout.reel_head=fread(fid,[80 40],'uchar').';
else,
  fseek(fid,3200,0);
end,

% load reel binary header
% -----------------------
dataout.reel_bin = zeros(30,1);
dataout.reel_bin(1:3)  =fread(fid,3,'int32');
dataout.reel_bin(4:27) =fread(fid,24,'int16');
fseek(fid,240,0);
dataout.reel_bin(28:30)=fread(fid,3,'int16'); 
%fseek(fid, pos,-1)
fseek(fid,94,0);

dataout.dt        =dataout.reel_bin(6)/1e6;
dataout.samples   =dataout.reel_bin(8);
dataout.dataformat=dataout.reel_bin(10);
if (dataout.dataformat==3) sample_bytes=2;
else                       sample_bytes=4;
end,
dataout.trace_num =(file_len-3600)/(240+dataout.samples*sample_bytes);

if ((dataout.dataformat<1) | (dataout.dataformat>5) | (dataout.dataformat==4)), 
  fclose(fid),
  error('Unrecognized data format'), 
end,

% load only a subset of data
% --------------------------
if isempty(traces), Tstart=1; Tend=dataout.trace_num;
else,               
  Tstart=max(traces(1),1); Tend=min(traces(2),dataout.trace_num);
  dataout.subset_limits=[Tstart Tend];                
end,
dataout.trace_num=Tend-Tstart+1;

if (~isempty(findstr(select,'tr_head')) ),
  dataout.tr_head=int32(zeros(76,dataout.trace_num));
  tr_h=1;
end,
if (~isempty(findstr(select,'tr_data')) ),
  dataout.tr_data=single(zeros(dataout.samples,dataout.trace_num));
  tr_d=1;
end,
if (~isempty(findstr(select,'tr_minmax')) ),
  dataout.tr_minmax=zeros(2,dataout.trace_num);
  tr_m=1;
end,
if (~isempty(findstr(select,'geometry')) ),
  dataout.tr_geometry=zeros(4,dataout.trace_num);
  tr_g=1;
end,
if (~isempty(findstr(select,'binning')) ),
  dataout.tr_binning=zeros(2,dataout.trace_num);
  tr_b=1;
end,

if ((tr_d+tr_h+tr_g)==0), 
  fclose(fid),
  return, 
end,

% salta le tracce non richieste
% -----------------------------
fseek(fid,(Tstart-1)*(240+dataout.samples*sample_bytes),0);

for itr=1:dataout.trace_num,
    
  % load trace header
  % ----------------
  if (tr_h==1),
    dataout.tr_head(1:7,itr)  =int32(fread(fid,[7,1],'int32'));
    dataout.tr_head(8:11,itr) =int32(fread(fid,[4,1],'int16'));  % 8: Trace identification code
    dataout.tr_head(12:19,itr)=int32(fread(fid,[8,1],'int32'));
    dataout.tr_head(20:21,itr)=int32(fread(fid,[2,1],'int16'));
    dataout.tr_head(22:25,itr)=int32(fread(fid,[4,1],'int32'));  % 22->25: Source X,Y Rec X,Y
    dataout.tr_head(26:71,itr)=int32(fread(fid,[46,1],'int16')); % 26:Coord. units 39:Num of samples 40: sampling interval
    dataout.tr_head(72:76,itr)=int32(fread(fid,[5,1],'int32'));  % 72-73: CDP X,Y
    fseek(fid,24,0);
    dataout.tr_head(77:78,itr)=int32(fread(fid,[2,1],'int32'));  % 77-78: InLine, XLine
    fseek(fid,8,0);
    if (tr_g==1), 
      dataout.tr_geometry(:,itr)=double(dataout.tr_head(22:25,itr));
    end,
    if (tr_b==1), 
      dataout.tr_binning(:,itr)=double(dataout.tr_head(77:78,itr));
    end,
  elseif (tr_g==1 | tr_b==1),
    if (tr_g==1),
      fseek(fid,72,0);
      dataout.tr_geometry(:,itr)=fread(fid,[4,1],'int32');  % byte 73->88: Source X,Y Rec X,Y
    end,
    if (tr_b==1),
      if (tr_g==1), 
        fseek(fid,136,0);
      else
        fseek(fid,224,0);
      end,
      dataout.tr_binning(:,itr)=fread(fid,[2,1],'int32');  % byte 225->232: InLine, XLine
      fseek(fid,8,0);
    else,
      fseek(fid,152,0);
    end,
  else,          
    fseek(fid,240,0);
  end,

  % load trace data
  % ---------------
  if (tr_d==1 | tr_m==1),
    if (dataout.dataformat==5),     % ieee big endian float
      this_trace=fread(fid,[dataout.samples,1],'float');
    elseif (dataout.dataformat==1), % ibm float
      data=fread(fid,[4,dataout.samples],'uchar');
  	  mant=data(2,:)/256+data(3,:)/65536+data(4,:)/16777216;
	    expo=bitand(data(1,:),127) - 64;
      signum=-ones(1,dataout.samples);
      signum(find(bitand(data(1,:),128)==0))=1;
  	  this_trace=(signum.*mant.*16.^expo).'; 	
    elseif (dataout.dataformat==2), % int 32
      this_trace=fread(fid,[dataout.samples,1],'int32');
    elseif (dataout.dataformat==3), % int 32
      this_trace=fread(fid,[dataout.samples,1],'int16');
    end,
    if (tr_d==1), dataout.tr_data(:,itr)  =single(this_trace); end,
    if (tr_m==1), dataout.tr_minmax(:,itr)=[min(this_trace),max(this_trace)].'; end,  
  else,
    fseek(fid,dataout.samples*sample_bytes,0);
  end,

end,  

fclose(fid);
return,
