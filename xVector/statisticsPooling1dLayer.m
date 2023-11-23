classdef statisticsPooling1dLayer < nnet.layer.Layer & nnet.layer.Formattable
    % This class is only for use in this example. It may be changed or
    % removed in a future release. 

    methods
        function this = statisticsPooling1dLayer(options)
            arguments
                options.Name = ''
            end
            this.Name = options.Name;
        end
        
        function X = predict(~, X)
            X = dlarray(stripdims([mean(X,3);std(X,0,3)]),"CB");
        end
        function X = forward(~, X)
            X = X + 0.0001*rand(size(X),"single");
            X = dlarray(stripdims([mean(X,3);std(X,0,3)]),"CB");
        end
    end
    
end