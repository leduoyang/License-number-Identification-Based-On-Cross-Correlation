# LICENSE NUMBER IDENTIFICATION 
Character Recognition based on cross correlation

## Workflow
```
Step 1. Binarization
Step 2. Connected component labeling
Step 3. Remove non-character components
Step 4. Sort the components by the pixel indexes
Step 5. Extract components with bounding box
Step 6. Calculate the number of holes for each components
Step 7. Recognition by cross correlation / total transfer distance computation
```

## Example
### Binarization
![alt text](https://github.com/leduoyang/Character-Recognition-based-on-cross-correlation/blob/master/img/1.png)
### Connected component labeling~Extract components with bounding box
![alt text](https://github.com/leduoyang/Character-Recognition-based-on-cross-correlation/blob/master/img/2.png)
### Recognition by cross correlation / total transfer distance computation
![alt text](https://github.com/leduoyang/Character-Recognition-based-on-cross-correlation/blob/master/img/3.png)
