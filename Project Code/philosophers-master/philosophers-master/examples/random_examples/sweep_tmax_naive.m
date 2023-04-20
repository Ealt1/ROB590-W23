clear;clc;close all;
%% User Defined Parameters
num_iteration = 2;1000;
random_number = 1:10;[1 7];% 7 9 10];
TMAX = 2;2:2:20;
Makespan = zeros(num_iteration,max(random_number), max(TMAX));
AllTimes = cell(num_iteration,max(random_number), max(TMAX));
for tmax = TMAX
    for r = random_number
        rng(0);
        r
        eval(['random', num2str(r), '_modified']);
%         prob_succ = parameters(:,5);
        %%
        prob_succ = 1/tmax + (1-1/tmax)*rand(35,1);
        %%
        % simulate
        for i = 1:num_iteration
            [time_elapsed, num_messages] = philosopher(Paths, prob_succ,'pre_load', ['rand', num2str(r), '_naive']);%, 'plot_stuff', ws); %, 'pre_load', );%, 'plot_stuff', ws);%, );
    %         [time_elapsed, num_messages] = philosopher(Paths, prob_succ,'pre_load', ['rand', num2str(r), '_', cycle_method], 'fix_seed', this_seed); %, 'pre_load', );%, 'plot_stuff', ws);%, );
            Makespan(i, r, tmax) = max(time_elapsed);
            AllTimes{i, r, tmax} = time_elapsed;
        end
%         save('Philosopher_naive_results_sweep_tmax')
    end
    
%     avg_makespan = zeros(10,1);
%     confidence_int = zeros(10,1);
%     for i = random_number
%     mi = find(Makespan(:,i)>0);
%     mi = Makespan(mi,i);
%     mydist = fitdist(mi, 'normal');
%     avg_makespan(i) = mean(mi);
%     a = mydist.paramci(0.05);
%     confidence_int(i) = mydist.mu - a(1);
%     end
end
1;
