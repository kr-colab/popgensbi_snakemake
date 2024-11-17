BEGIN {FS="\t"; OFS="\t"}
/^#/ {print $0; next} 
{$8="AA=" $4; print $0}