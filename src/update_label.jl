"""
    update_label!(label)

- label is a vector of integers with possibly value 0

if there are m different positive integers, set f, a bijective map between 
these integers and 1:m ; and f(0)=0
return f(label).
"""
function update_label!(label)

    sort_lab = sortperm(label)

    new_sort_label = zero(label)
    new_lab = 1
    lab = label[sort_lab[1]]
    new_sort_label[1] = new_lab
    for i = 2:length(label)
        if label[sort_lab[i]] != lab
            new_lab = new_lab + 1
            lab = label[sort_lab[i]]
        end
        new_sort_label[i] = new_lab
    end
    new_label = zero(label)
    for i in eachindex(label)
        new_label[sort_lab[i]] = new_sort_label[i]
    end

    return new_label

end
