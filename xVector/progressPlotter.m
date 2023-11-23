classdef progressPlotter < handle
% This class is only for use in this example. It may be changed or removed
% in a future release.
    
    properties
        start
        myFigure
        accuracyPlot
        lineAccuracyTrain
        lineAccuracyValidation
        lossPlot
        lineLossTrain
        lineLossValidation
        Epoch = 1
        classes
    end
    
    methods
        function obj = progressPlotter(classes)
            obj.classes = classes;
            obj.start = tic;
            
            obj.myFigure = figure;
            
            tiledlayout(2,1);
            
            obj.accuracyPlot = nexttile;
            obj.lineAccuracyTrain = animatedline('Color',[0 0.4470 0.7410]);
            obj.lineAccuracyValidation = animatedline('Color','k','LineStyle','--','Marker','o');
            legend({'Train','Validation'},'AutoUpdate','off','Location','southeast')
            xlabel("Iteration")
            ylabel("Accuracy (%)")
            grid on
            
            obj.lossPlot = nexttile;
            obj.lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
            obj.lineLossValidation = animatedline('Color','k','LineStyle','--','Marker','o');
            legend({'Train','Validation'},'AutoUpdate','off','Location','northeast')
            ylim([0 inf])
            xlabel("Iteration")
            ylabel("Loss")
            grid on
        end
        
        function updateTrainingProgress(obj,nvargs)
            arguments
                obj
                nvargs.Epoch
                nvargs.LearnRate
                nvargs.Predictions
                nvargs.Targets
                nvargs.Iteration
                nvargs.Loss
            end
            set(0,'CurrentFigure',obj.myFigure)
            
            D = duration(0,0,toc(obj.start),'Format','hh:mm:ss');
            title(obj.accuracyPlot,"Epoch: " + nvargs.Epoch + ", Elapsed: " + string(D) + ", Learn Rate: " + string(nvargs.LearnRate));
            
            predictedLabel = onehotdecode(extractdata(nvargs.Predictions),obj.classes,1);
            trueLabel = onehotdecode(nvargs.Targets,obj.classes,1);
            accuracy = mean(trueLabel==predictedLabel)*100;
            
            addpoints(obj.lineAccuracyTrain,nvargs.Iteration,accuracy);
            
            addpoints(obj.lineLossTrain,nvargs.Iteration,nvargs.Loss);
            
            if nvargs.Epoch ~= obj.Epoch
                xline(obj.accuracyPlot,nvargs.Iteration-1,'k:')
                xline(obj.lossPlot,nvargs.Iteration-1,'k:')
                obj.Epoch = nvargs.Epoch;
            end
            
            drawnow
        end
        function updateValidation(obj,nvargs)
            arguments
                obj
                nvargs.Predictions
                nvargs.Targets
                nvargs.Iteration
            end
            set(0,'CurrentFigure',obj.myFigure)
            
            a = onehotencode(nvargs.Predictions,1,'ClassNames',obj.classes);
            b = onehotencode(nvargs.Targets,1,'ClassNames',obj.classes);

            validationLoss = crossentropy(dlarray(a),dlarray(b),'DataFormat','CB');

            accuracy = mean(nvargs.Targets==nvargs.Predictions)*100;
            
            addpoints(obj.lineAccuracyValidation,nvargs.Iteration,accuracy);

            addpoints(obj.lineLossValidation,nvargs.Iteration,double(gather(extractdata(validationLoss))))
            
            drawnow
        end
    end
end
