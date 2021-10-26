"""
    update_label!(label)

- label is a vector of integers with possibly value 0
if there are m different positive integers, set f, a bijective map between these integers and 1:m ; and f(0)=0
return f(label).
"""
function update_label!(label)

    sort_lab = sort(label, index.return = TRUE)

    new_sort_label = rep(0,length(label))
    new_lab = 1
    lab = sort_lab$x[1]
    new_sort_label[1] = new_lab
    for(i in 2:length(label)){
      if(sort_lab$x[i]!=lab)
        new_lab = new_lab + 1
        lab = sort_lab$x[i]
      end
      new_sort_label[i] = new_lab
    end
    new_label = rep(0,length(label))
    for(i in 1:length(label))
      new_label[sort_lab$ix[i]] = new_sort_label[i]
    end
    return(new_label)

end

