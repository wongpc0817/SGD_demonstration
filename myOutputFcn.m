function stop = myOutputFcn(app,info)
    persistent hFig hAxesLoss hLineLoss hTextLoss hAxesAcc hLineAcc hTextAcc  ymax ...
        hAxesLoss1 hLineLoss1 hTextLoss1 hAxesAcc1 hLineAcc1 hTextAcc1 ymax1...
        bgColor1 bgColor validationCount prevIter

    stop = false;
    
    if info.State == "start"
        

        % Create a new figure at the beginning of the training
        hFig = figure('Name', 'Training Progress', 'NumberTitle', 'off');
        ymax = 0;
        % Create axes and line for the training loss plot
        hAxesLoss = subplot(2, 2, 1, 'Parent', hFig);
        hLineLoss = animatedline(hAxesLoss, 'Color', 'b', 'LineWidth', 1);

        

        xlabel(hAxesLoss, 'Iteration');
        ylabel(hAxesLoss, 'Training Loss');
        title(hAxesLoss, 'Training Loss vs Iteration');

        % Create text for the training loss value
        hTextLoss = text(hAxesLoss, 0.02, 0.95, 'Loss: N/A', ...
            'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold');

        % Create axes and line for the training accuracy plot
        hAxesAcc = subplot(2, 2, 2, 'Parent', hFig);
        hLineAcc = animatedline(hAxesAcc, 'Color', 'b', 'LineWidth', 1);
        
        xlabel(hAxesAcc, 'Iteration');
        ylabel(hAxesAcc, 'Training Accuracy (%)');
        title(hAxesAcc, 'Training Accuracy vs Iteration');

        % Create text for the training accuracy value
        hTextAcc = text(hAxesAcc, 0.02, 0.95, 'Accuracy: N/A', ...
            'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold');
        validationCount=0;
        


        %%%%%%%%%%%%%%%%%%%%%%
        ymax1 = 0;
        % Create axes and line for the training loss plot
        hAxesLoss1 = subplot(2, 2, 3, 'Parent', hFig);
        hLineLoss1 = animatedline(hAxesLoss1, 'Color', 'r', 'LineWidth', 1);

        

        xlabel(hAxesLoss1, 'Iteration');
        ylabel(hAxesLoss1, 'Validation Loss');
        title(hAxesLoss1, 'Validation Loss vs Iteration');

        % Create text for the training loss value
        hTextLoss1 = text(hAxesLoss1, 0.02, 0.95, 'Loss: N/A', ...
            'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold');

        % Create axes and line for the training accuracy plot
        hAxesAcc1 = subplot(2, 2, 4, 'Parent', hFig);
        hLineAcc1 = animatedline(hAxesAcc1, 'Color', 'r', 'LineWidth', 1);
        
        xlabel(hAxesAcc1, 'Iteration');
        ylabel(hAxesAcc1, 'Validation Accuracy (%)');
        title(hAxesAcc1, 'Validation Accuracy vs Iteration');

        % Create text for the training accuracy value
        hTextAcc1 = text(hAxesAcc1, 0.02, 0.95, 'Accuracy: N/A', ...
            'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold');
        prevIter = 0;


    elseif info.State == "iteration"

        if mod(info.Epoch, 2) == 0
            bgColor = [186, 247, 255]/255; 
            bgColor1 = [252, 189, 255]/255;
        else
            bgColor= [76, 194, 255]/255; 
            bgColor1 = [255, 140, 202]/255;
        end

        if info.ValidationLoss + 1 >= ymax1
            ymax1 = info.ValidationLoss + 1;
        end

        % Update the training loss plot
        if ~isempty(info.ValidationLoss)
            validationCount=validationCount+1;
%             if mod(validationCount,2)==0
%                 bgColor = [252, 189, 255]/255;
%             else
%                 bgColor = [255, 140, 202]/255;
%             end
            addpoints(hLineLoss1, info.Iteration, info.ValidationLoss);
            set(hTextLoss1, 'String', sprintf('Loss: %.4f', info.ValidationLoss));
    
            % Update the training accuracy plot
            addpoints(hLineAcc1, info.Iteration, info.ValidationAccuracy);
            set(hTextAcc1, 'String', sprintf('Accuracy: %.2f', info.ValidationAccuracy));
    
            hAxesLoss1.YLim = [0, ymax1];
            app.currentAccVal(end+1)=info.ValidationAccuracy;
            app.currentLossVal(end+1)=info.ValidationLoss;
            app.currentValidationTimeStamps(end+1)=info.Iteration;
            prevIter = info.Iteration;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if info.TrainingLoss + 1 >= ymax
            ymax = info.TrainingLoss + 1;
        end

        % Update the training loss plot
        addpoints(hLineLoss, info.Iteration, info.TrainingLoss);
        set(hTextLoss, 'String', sprintf('Loss: %.4f', info.TrainingLoss));

        % Update the training accuracy plot
        addpoints(hLineAcc, info.Iteration, info.TrainingAccuracy);
        set(hTextAcc, 'String', sprintf('Accuracy: %.2f', info.TrainingAccuracy));

        hAxesLoss.YLim = [0, ymax];

        
        
        drawStart = info.Iteration;
        drawEnd = info.Iteration+1;
        patch(hAxesLoss, [drawStart, drawStart, drawEnd, drawEnd], ...
            [0,ymax,ymax,0], bgColor, 'EdgeColor', 'none', 'FaceAlpha', 0.1);
        patch(hAxesAcc,[drawStart, drawStart, drawEnd, drawEnd], ...
            [0,100,100,0],bgColor, 'EdgeColor', 'none', 'FaceAlpha', 0.1);
        patch(hAxesLoss1, [drawStart, drawStart, drawEnd, drawEnd], ...
            [0,ymax,ymax,0], bgColor1, 'EdgeColor', 'none', 'FaceAlpha', 0.1);
        patch(hAxesAcc1,[drawStart, drawStart, drawEnd, drawEnd], ...
            [0,100,100,0],bgColor1, 'EdgeColor', 'none', 'FaceAlpha', 0.1);
        
        app.currentAcc(end+1)=info.TrainingAccuracy;
        app.currentLoss(end+1)=info.TrainingLoss;
        app.currentTimeStamps(end+1)=info.Iteration;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        drawnow;  
        
    elseif info.State == "done"

        hAxesLoss.XLim=[0,info.Iteration+2];
        hAxesAcc.XLim=[0,info.Iteration+2];
        hAxesAcc1.XLim=[0,info.Iteration+2];
        hAxesLoss1.XLim=[0,info.Iteration+2];
        patch(hAxesLoss1, [info.Iteration, info.Iteration, info.Iteration+2, info.Iteration+2], ...
                [0,ymax1,ymax1,0], bgColor1, 'EdgeColor', 'none', 'FaceAlpha', 0.1);
        patch(hAxesAcc1, [info.Iteration, info.Iteration, info.Iteration+2, info.Iteration+2], ...
            [0,100,100,0],bgColor1, 'EdgeColor', 'none', 'FaceAlpha', 0.1);
        patch(hAxesLoss, [info.Iteration, info.Iteration, info.Iteration+2, info.Iteration+2], ...
            [0,ymax,ymax,0], bgColor, 'EdgeColor', 'none', 'FaceAlpha', 0.1);
        patch(hAxesAcc, [info.Iteration, info.Iteration, info.Iteration+2, info.Iteration+2], ...
            [0,100,100,0],bgColor, 'EdgeColor', 'none', 'FaceAlpha', 0.1);
        if ~isempty(app.historyLoss)
            hold(hAxesLoss,"on");
            plot(hAxesLoss,0:length(app.historyLoss)-1,app.historyLoss,...
                'Color','#808080')
            hold(hAxesLoss,"off")
        end
        if ~isempty(app.historyAcc)
            hold(hAxesAcc,"on");
            plot(hAxesAcc,0:length(app.historyAcc)-1,app.historyAcc,...
                'Color','#808080')
            hold(hAxesAcc,'off')
        end
        if ~isempty(app.historyLossVal)
            
            hold(hAxesLoss1,"on");
            plot(hAxesLoss1,app.historyValidationTimeStamps,app.historyLossVal,...
                'Color','#808080')
            hold(hAxesLoss1,"off")
        end
        if ~isempty(app.historyAccVal)
            hold(hAxesAcc1,"on");
            plot(hAxesAcc1,app.historyValidationTimeStamps,app.historyAccVal,...
                'Color','#808080')
            hold(hAxesAcc1,'off')
        end
        
        % Clear the persistent variables at the end of the training
        hFig = [];
        hAxesLoss = [];
        hLineLoss = [];
        hTextLoss = [];
        hAxesAcc = [];
        hLineAcc = [];
        hTextAcc = [];
        
        app.historyAcc=app.currentAcc;
        app.historyLoss=app.currentLoss;
        app.currentAcc=[];
        app.currentLoss=[];

        %%%%%%%%%%%%%%%%%%%%

        
        
        % Clear the persistent variables at the end of the training
        hAxesLoss1 = [];
        hLineLoss1 = [];
        hTextLoss1 = [];
        hAxesAcc1 = [];
        hLineAcc1 = [];
        hTextAcc1 = [];
        
        app.historyAccVal=app.currentAccVal;
        app.historyLossVal=app.currentLossVal;
        app.currentAccVal=[];
        app.currentLossVal=[];

        app.historyValidationTimeStamps=app.currentValidationTimeStamps;
        app.historyTimeStamps=app.currentTimeStamps;
        app.currentTimeStamps=[];
        app.currentValidationTimeStamps=[];
    end
end